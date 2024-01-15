#include "OcrLite.h"
#include "OcrUtils.h"
#include <stdarg.h> //windows&linux


OcrLite::OcrLite() {}

OcrLite::~OcrLite()
{
	if (isOutputResultTxt)
	{
		fclose(resultTxt);
	}
}

void OcrLite::setNumThread(int numOfThread)
{
	dbNet.setNumThread(numOfThread);
	angleNet.setNumThread(numOfThread);
	crnnNet.setNumThread(numOfThread);
}

void OcrLite::initModels(
	const void* det_model_data, size_t det_model_data_len,
	const void* cls_model_data, size_t cls_model_data_len,
	const void* rec_model_data, size_t rec_model_data_len,
	std::string keys_data)
{

	dbNet.initModel(det_model_data, det_model_data_len);
	angleNet.initModel(cls_model_data, cls_model_data_len);
	crnnNet.initModel(rec_model_data, rec_model_data_len, keys_data);
}


cv::Mat makePadding(cv::Mat& src, const int padding)
{
	if (padding <= 0) return src;
	cv::Scalar paddingScalar = { 255, 255, 255 };
	cv::Mat paddingSrc;
	cv::copyMakeBorder(src, paddingSrc, padding, padding, padding, padding, cv::BORDER_ISOLATED, paddingScalar);
	return paddingSrc;
}


std::vector<cv::Mat> OcrLite::getPartImages(cv::Mat& src, std::vector<TextBox>& textBoxes)
{
	std::vector<cv::Mat> partImages;
	for (int i = 0; i < textBoxes.size(); ++i) {
		cv::Mat partImg = getRotateCropImage(src, textBoxes[i].boxPoint);
		partImages.emplace_back(partImg);
	}
	return partImages;
}

std::string OcrLite::detect(cv::Mat img, const int padding, const int maxSideLen, float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle)
{
	int originMaxSide = (std::max)(img.cols, img.rows);
	int resize;
	if (maxSideLen <= 0 || maxSideLen > originMaxSide)
	{
		resize = originMaxSide;
	}
	else
	{
		resize = maxSideLen;
	}
	resize += 2 * padding;
	cv::Rect paddingRect(padding, padding, img.cols, img.rows);
	cv::Mat paddingSrc = makePadding(img, padding);
	ScaleParam scale = getScaleParam(paddingSrc, resize);

	int thickness = getThickness(paddingSrc);
	std::vector<TextBox> textBoxes = dbNet.getTextBoxes(paddingSrc, scale, boxScoreThresh, boxThresh, unClipRatio);
	drawTextBoxes(paddingSrc, textBoxes, thickness);

	//---------- getPartImages ----------
	std::vector<cv::Mat> partImages = getPartImages(paddingSrc, textBoxes);

	//Logger("---------- step: angleNet getAngles ----------\n");
	std::vector<Angle> angles;
	angles = angleNet.getAngles(partImages, doAngle, mostAngle);

	//Rotate partImgs
	for (int i = 0; i < partImages.size(); ++i)
	{
		if (angles[i].index == 1)
		{
			partImages.at(i) = matRotateClockWise180(partImages[i]);
		}
	}
	std::vector<TextLine> textLines = crnnNet.getTextLines(partImages);
	// TextLines
	int kuos = paddingRect.x;//padding conversion
	int x1, y1, x2, y2;
	std::string txttxt;
	for (int i = 0; i < textLines.size(); ++i)
	{
		std::ostringstream txtScores;
		for (int s = 0; s < textLines[i].charScores.size(); ++s)
		{
			if (s == 0)
			{
				txtScores << textLines[i].charScores[s];
			}
			else
			{
				txtScores << "," << textLines[i].charScores[s];
			}
		}

		if (textLines[i].text.length() > 0)
		{
			//文本内容
			txttxt += textLines[i].text.c_str();
			txttxt += "|#|";

			//文本位置
			x1 = std::min(textBoxes[i].boxPoint[0].x, textBoxes[i].boxPoint[1].x);
			x1 = std::min(x1, textBoxes[i].boxPoint[2].x);
			x1 = std::min(x1, textBoxes[i].boxPoint[3].x) - kuos;

			y1 = std::min(textBoxes[i].boxPoint[0].y, textBoxes[i].boxPoint[1].y);
			y1 = std::min(y1, textBoxes[i].boxPoint[2].y);
			y1 = std::min(y1, textBoxes[i].boxPoint[3].y) - kuos;


			x2 = std::max(textBoxes[i].boxPoint[0].x, textBoxes[i].boxPoint[1].x);
			x2 = std::max(x2, textBoxes[i].boxPoint[2].x);
			x2 = std::max(x2, textBoxes[i].boxPoint[3].x) - kuos - x1;

			y2 = std::max(textBoxes[i].boxPoint[0].y, textBoxes[i].boxPoint[1].y);
			y2 = std::max(y2, textBoxes[i].boxPoint[2].y);
			y2 = std::max(y2, textBoxes[i].boxPoint[3].y) - kuos - y1;

			txttxt += std::to_string(x1) + ",";
			txttxt += std::to_string(y1) + ",";
			txttxt += std::to_string(x2) + ",";
			txttxt += std::to_string(y2) + ",";


			txttxt += std::to_string(textBoxes[i].boxPoint[0].x - kuos) + "," + std::to_string(textBoxes[i].boxPoint[0].y - kuos) + ",";
			txttxt += std::to_string(textBoxes[i].boxPoint[1].x - kuos) + "," + std::to_string(textBoxes[i].boxPoint[1].y - kuos) + ",";
			txttxt += std::to_string(textBoxes[i].boxPoint[2].x - kuos) + "," + std::to_string(textBoxes[i].boxPoint[2].y - kuos) + ",";
			txttxt += std::to_string(textBoxes[i].boxPoint[3].x - kuos) + "," + std::to_string(textBoxes[i].boxPoint[3].y - kuos); //+ ",";

			txttxt += "|#|";

			//文本可信度
			txttxt += txtScores.str();
			txttxt += "\r\n";
		}
	}
	return txttxt;
}