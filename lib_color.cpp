
#include "lib_color.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

/*
* class that includes some color methods          
*  
*                                                       
* written by Cagatay Odabasi                            
*
*/

using namespace std;
using namespace cv;

vector<Mat> Colorful::extractRGB(Mat im)
{
	vector<Mat> channels; // create vector that contains RGB channels
	 
    split(im, channels);  // split multiple channels

	return channels;
}

// method to convert BGR to HSV
// and return each of HSV channels individually
std::vector<Mat> Colorful::convertBGR2HSV(cv::Mat bgr)
{
	vector<Mat> hsv_channels(3);

	Mat hsv;

	//bgr *= 1/255;

	cvtColor(bgr, hsv, CV_BGR2HSV);

	split(hsv, hsv_channels);

	return hsv_channels;
}

// method to convert HSV to BGR
// and returns the each BGR channel seperately
std::vector<Mat> Colorful::convertHSV2BGR(cv::Mat hsv)
{
	vector<Mat> bgr_channels(3);

	Mat bgr;

	//bgr *= 1/255;

	cvtColor(hsv, bgr, CV_HSV2BGR);

	split(bgr, bgr_channels);

	return bgr_channels;
}
