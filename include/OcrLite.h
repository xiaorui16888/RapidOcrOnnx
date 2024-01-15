#ifndef __OCR_LITE_H__
#define __OCR_LITE_H__

#include "opencv2/core.hpp"
#include "OcrStruct.h"
#include "DbNet.h"
#include "AngleNet.h"
#include "CrnnNet.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
class OcrLite {
public:
    OcrLite();

    ~OcrLite();

    void setNumThread(int numOfThread);
    //void initModels(const std::string& detPath, const std::string& clsPath,const std::string& recPath, const std::string& keysPath);
    void initModels(
        const void* det_data, size_t det_data_len,
        const void* cls_data, size_t cls_data_len,
        const void* rec_data, size_t rec_data_len,
        std::string keys_data);

    std::string detect(cv::Mat img, int padding, int maxSideLen, float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle);

private:
    bool isOutputConsole = false;
    bool isOutputPartImg = false;
    bool isOutputResultTxt = false;
    bool isOutputResultImg = false;
    FILE* resultTxt;
    DbNet dbNet;
    AngleNet angleNet;
    CrnnNet crnnNet;

    std::vector<cv::Mat> getPartImages(cv::Mat& src, std::vector<TextBox>& textBoxes);
};

#endif //__OCR_LITE_H__
