#include <omp.h>
#include <cstdio>
#include <string>
#include "main.h"
#include "version.h"
#include "OcrLite.h"
#include "OcrUtils.h"

static OcrLite* ocrLite;


void _stdcall model(char* det_data, int det_data_len, const char* cls_data, int cls_data_len, const char* rec_data, int rec_data_len, char* keys_data, int numThread)
{
	delete ocrLite;
	ocrLite = nullptr;
	ocrLite = new OcrLite();

	omp_set_num_threads(numThread);
	ocrLite->setNumThread(numThread);
	ocrLite->initModels(det_data, det_data_len, cls_data, cls_data_len, rec_data, rec_data_len, keys_data);
}

char* _stdcall ocr(char* imgbuffer, int imgbufferlen, int padding, int maxSideLen, float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostangle)
{
	cv::Mat img_decode;
	std::vector<uchar> data;
	for (int i = 0; i < imgbufferlen; ++i)
	{
		data.push_back(imgbuffer[i]);
	}
	img_decode = cv::imdecode(data, cv::IMREAD_COLOR);

	if (!img_decode.data)
	{
		return "";
	}

	std::string result = ocrLite->detect(img_decode, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostangle);

	char* reschar = new char[strlen(result.c_str()) + 1];
	strcpy(reschar, result.c_str());
	return reschar;
}

void _stdcall del(char* buffer)
{
	delete buffer;
	buffer = nullptr;
}