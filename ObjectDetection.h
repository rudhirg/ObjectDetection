#ifndef OBJ_DETECT_H
#define OBH_DETECT_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>

#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <algorithm>

extern "C" {
#include "svm-light/svm_common.h"
#include "svm-light/svm_learn.h"
}

using namespace std;

class SVMModule;

void SplitStr(std::string str, char sp, std::vector<std::string>& splits);

class ObjectDetection {
public:
	ObjectDetection();
	~ObjectDetection() {}

	int Close();

	int trainMultiScale(std::string img, bool classType);
	int train(cv::Mat& img, bool classType);
	int InitTrain();
	
	int test(std::string imgName);	// uses detectMultiScale
	int testClassify(cv::Mat& img, double& prob);	// uses svm_classify to know if the image is pos/neg; works only on 64x64 image; also gives prob
	cv::Rect testClassifyMultiScale(cv::Mat& img, int sz, double& prob);

	int InitTest(std::string modelName);

	int GetHSVHistogram(cv::Mat& img, std::vector<float>& histVec);
	int GetHogDescriptor(cv::Mat& img, std::vector<float> &hogDesc);

	void DumpToSVMFormat(std::vector<float>& rhog, bool classType);

	void SetThreshold(double thresh) { m_threshold = thresh; }
	void SetObjectName(std::string name) { m_objectName = name; }
	void SetWindowSize(int ht, int wid) { m_winHt = ht; m_winWid = wid; }


private:
	std::ofstream		m_svmfile;
	MODEL*			m_ptrainModel;
	cv::HOGDescriptor*	m_phog;

	double			m_threshold; 	// only used while testing

	SVMModule*		m_pSvm;
	std::string		m_objectName;

	int			m_winHt;
	int			m_winWid;
};

class SVMModule {
public:
	SVMModule();
	~SVMModule() {}

	void SetThreshold(double thresh) { m_threshold = thresh; }

	int LoadSVMModelFile(std::string modelFile);
	int GetSupportVector(std::vector<float> &supVector);

	double classify(std::vector<float>& instance);

private:
	MODEL*			m_pTrainModel;

	double			m_threshold;
};

#endif
