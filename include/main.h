
#ifdef __cplusplus
#define EXPORT extern "C" __declspec (dllexport)
#else
#define EXPORT __declspec (dllexport)
#endif

EXPORT void _stdcall model(char* det_data, int det_data_len, const char* cls_data, int cls_data_len, const char* rec_data, int rec_data_len, char* keys_data, int numThread);
EXPORT char* _stdcall ocr(char* imgbuffer, int imgbufferlen, int padding, int maxSideLen, float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostangle);
EXPORT void _stdcall del(char* buffer);