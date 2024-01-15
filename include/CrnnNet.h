#ifndef __OCR_CRNNNET_H__
#define __OCR_CRNNNET_H__

#include "OcrStruct.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>

class CrnnNet {
public:

    CrnnNet();

    ~CrnnNet();

    void setNumThread(int numOfThread);

    void initModel(const void* model_data, size_t model_data_length, std::string&keys_data);

    std::vector<TextLine> getTextLines(std::vector<cv::Mat> &partImg);

private:
    bool isOutputDebugImg = false;
    Ort::Session *session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "CrnnNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;

    char *inputName;
    char *outputName;

    const float meanValues[3] = {127.5, 127.5, 127.5};
    const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    //const int dstHeight = 32;
    const int dstHeight = 48;
    std::vector<std::string> keys;

    TextLine scoreToTextLine(const std::vector<float> &outputData, int h, int w);

    TextLine getTextLine(const cv::Mat &src);
};


#endif //__OCR_CRNNNET_H__
