
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>

/*

Library that stores the features methods 

Cagatay Odabasi

*/

class FeatureProc 
{

	public:
	// Empty Constructor 
	FeatureProc(void) {};

	// Harris Detector Method 
	std::vector<cv::Point> harrisDetector(cv::Mat, int);

	// SIFT Detector Method
	std::vector<cv::KeyPoint> siftFind(cv::Mat);

	// the method applies BoW algorithm
	void siftBOW(std::vector<cv::Mat>, std::vector<cv::Mat>);

	//method that visualize histogram
	void FeatureProc::showHistogram(cv::Mat& img);

};