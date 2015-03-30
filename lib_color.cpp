
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

// method that limits the hue value between a range
cv::Mat Colorful::limitHue(cv::Mat hsv, cv::Mat bgr, float low_lim, float high_lim)
{
	// Input
	// hsv is a Mat that contains image
	// bgr is a Mat that contains BGR version of image
	// low_lim is the lower limit of hue
	// high_lim is the higher limit of hue
	//
	// Output
	// image whose hue value is limited

	Mat limited_im = Mat::zeros(576,480,CV_32F);

	// Split the HSV channels 
	vector<Mat> channels;

	split(hsv, channels);

	for(int i = 0; i < hsv.rows; i++)
	{
		for (int j = 0; j < hsv.cols; j++)
		{
			float value = channels.at(0).at<float>(i,j);			

			if ((value < high_lim)&&(value > low_lim))
			{			
				limited_im.at<float>(i,j) = bgr.at<float>(i,j);
			}
		}
	}

	return limited_im;
}