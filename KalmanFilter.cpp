#include "KalmanFilter.h"

Kalman::Kalman() {
	m_pkalman = new cv::KalmanFilter();
	m_numMeasureDim = 3;
	m_numStatesDim = 6;

	m_state = new cv::Mat(m_numStatesDim, 1, CV_32F);
	m_pProcessNoise = new cv::Mat(m_numStatesDim, 1, CV_32F);
	m_pMeasurementNoise = new cv::Mat(m_numMeasureDim, 1, CV_32F);

	m_measurement = new cv::Mat(m_numMeasureDim, 1, CV_32F);
}

void Kalman::Init() {
	m_bprior = true;
	m_pkalman->init(m_numStatesDim, m_numMeasureDim, 0);
	(*m_measurement) = cv::Mat::zeros(m_numMeasureDim, 1, CV_32F);

	const float A[] = {
		1,0,0,1,0,0,  // x + dx
		0,1,0,0,1,0, // y +dy
		0,0,1,0,0,1, // z +dz
		0,0,0,1,0,0,  // dy =dy
		0,0,0,0,1,0,  // dy =dy
		0,0,0,0,0,1  // dy =dy
	};

	m_pkalman->transitionMatrix = *(cv::Mat_<float>(m_numStatesDim, m_numStatesDim) << 
		1,0,0,1,0,0,  // x + dx
		0,1,0,0,1,0, // y +dy
		0,0,1,0,0,1, // z +dz
		0,0,0,1,0,0,  // dy =dy
		0,0,0,0,1,0,  // dy =dy
		0,0,0,0,0,1);  // dy =dy

	setIdentity(m_pkalman->measurementMatrix);
	setIdentity(m_pkalman->processNoiseCov, cv::Scalar::all(0.004));
	setIdentity(m_pkalman->measurementNoiseCov, cv::Scalar::all(0.004));
	setIdentity(m_pkalman->errorCovPost, cv::Scalar::all(1));

}

void Kalman::InitPrediction(pcl::PointXYZ* leftPt, pcl::PointXYZ* rhtPt)
{
	m_bprior = false;
	m_pkalman->statePost.at<float>(0) = (leftPt->x + rhtPt->x)/2.0;
	m_pkalman->statePost.at<float>(1) = (leftPt->y + rhtPt->y)/2.0;
	m_pkalman->statePost.at<float>(2) = (leftPt->z + rhtPt->z)/2.0;
}

pcl::PointXYZ Kalman::Predict() {
	cv::Mat prediction = m_pkalman->predict();
	pcl::PointXYZ predictedPt;
	predictedPt.x = prediction.at<float>(0);
	predictedPt.y = prediction.at<float>(1);
	predictedPt.z = prediction.at<float>(2);

	prediction.copyTo(*m_state);

	return predictedPt;
}

pcl::PointXYZ Kalman::Correct(pcl::PointXYZ* leftPt, pcl::PointXYZ* rhtPt, bool bDetections) {

	pcl::PointXYZ predictedPt;
	if(m_bprior && bDetections) {
		this->InitPrediction(leftPt, rhtPt);
		//return 0;
		predictedPt.x = (leftPt->x + rhtPt->x)/2.0;
		predictedPt.y = (leftPt->y + rhtPt->y)/2.0;
		predictedPt.z = (leftPt->z + rhtPt->z)/2.0;
		return predictedPt;
	}
	
	cv::Mat prediction = m_pkalman->predict();
	predictedPt.x = prediction.at<float>(0);
	predictedPt.y = prediction.at<float>(1);
	predictedPt.z = prediction.at<float>(2);

	// since new detections are available, update the new positions
	if(bDetections) {
		m_state->at<float>(0) = (leftPt->x + rhtPt->x)/2.0;
		m_state->at<float>(1) = (leftPt->y + rhtPt->y)/2.0;
		m_state->at<float>(2) = (leftPt->z + rhtPt->z)/2.0;
	} else {
		m_state->at<float>(0) = predictedPt.x;
		m_state->at<float>(1) = predictedPt.y;
		m_state->at<float>(2) = predictedPt.z;
	}

	randn((*m_measurement), cv::Scalar::all(0), cv::Scalar::all(m_pkalman->measurementNoiseCov.at<float>(0)));

	// zk = hk*xk + vk
	(*m_measurement) += m_pkalman->measurementMatrix * (*m_state);

	// adjust kalman
	m_pkalman->correct((*m_measurement));

	// calculate new state
	(*m_state) = m_pkalman->transitionMatrix * (*m_state) + (*m_pProcessNoise);

	if(bDetections == false)
		printf("Kalman Predictions - Predicted(x,y,z): %0.3f - %0.3f - %0.3f\n", predictedPt.x, predictedPt.y, predictedPt.z);
	else
		printf("Kalman Predictions - Predicted(x,y,z): %0.3f - %0.3f - %0.3f, Observation: %0.3f - %0.3f - %0.3f\n", predictedPt.x, predictedPt.y, predictedPt.z, 
								m_state->at<float>(0), m_state->at<float>(1), m_state->at<float>(2));

	return predictedPt;
}
