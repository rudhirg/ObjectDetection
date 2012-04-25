#ifndef KALMAN_H
#define KALMAN_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Kalman {

public:
	Kalman();
	~Kalman() {}

	//KalmanFilter KF(6, 1, 0);
	cv::KalmanFilter* 			m_pkalman;
	cv::Mat*				m_pProcessNoise;
	cv::Mat*				m_pMeasurementNoise;
	cv::Mat*				m_state;
	cv::Mat*				m_measurement;

	bool					m_bMeasurementAvail;
	bool					m_bprior;	// first time detection 

	int					m_numStatesDim;
	int					m_numMeasureDim;

	
	void Init();
	void InitPrediction(pcl::PointXYZ* leftPt, pcl::PointXYZ* rhtPt);
	pcl::PointXYZ Predict();
	pcl::PointXYZ Correct(pcl::PointXYZ* leftPt, pcl::PointXYZ* rhtPt, bool bD);
};

#endif
