#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\flann\miniflann.hpp>
#include <iostream>

#include "lib_features.h"

using namespace cv;


// Method to find corners and show them to user 
// Input
// img: Mat formatted input image (intensity channel)
// threshold: for harris corner detector
// Output
// the N point vector that contains corners 
vector<Point> FeatureProc::harrisDetector(Mat img, int threshold)
{
	//Variable Definitions
	Mat corners = Mat::zeros( img.size(), CV_32FC1 );
	Mat corners_normalized, corners_normalized_scaled, img_gray;

	// create a copy of image
	img_gray = img;

	// these variables taken from Original Tutorial of OpenCV
	int block_size = 2;
	int aperture_size = 3;
	double k = 0.04;

	// resulting vector
	vector<Point> corner_points_vec;

	/// Corner Detection
	cornerHarris( img, corners, block_size, aperture_size, k, BORDER_DEFAULT );

	/// Normalization
	normalize( corners, corners_normalized, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( corners_normalized, corners_normalized_scaled );

	/// Drawing circles
	for( int j = 0; j < corners_normalized.rows ; j++ )

	{ 
		for( int i = 0; i < corners_normalized.cols; i++ )
		{

			if( (int) corners_normalized.at<float>(j,i) > threshold )
			{
				circle(img_gray, Point( i, j ), 5,  Scalar(0, 0, 255));

				corner_points_vec.push_back(Point(i,j));
			}

		}
	}

	/// Showing the result
	namedWindow( "Corners", CV_WINDOW_AUTOSIZE );
	imshow( "Corners", img_gray );

	waitKey(0);

	// close all windows
	cvDestroyAllWindows();

	std::cout << std::endl << "there are " << corner_points_vec.size() << " corners detected" << std::endl;
	std::cout << corner_points_vec;


	return corner_points_vec;
}

// SIFT Detector Method
// the method that finds sift keypoints and prints them to user
// Input
// img in grayscale 
// Output 
// sift keypoints in vector format
// I got help for this code from http://stackoverflow.com/questions/24961136/opencv-sift-key-points-extraction-isuue
vector<KeyPoint> FeatureProc::siftFind(cv::Mat _img)
{
	Mat output, img_gray;

	// Create a copy of img
	img_gray = _img;

	// Define a sift detector
	SiftFeatureDetector detector;
	vector<KeyPoint> keypoints;

	// Detect the keypoints
	detector.detect(_img, keypoints);

	// write them on image
	drawKeypoints(img_gray, keypoints, output);

	// Show them to user
	namedWindow("SIFT");

	imshow("SIFT",output);

	waitKey(0);

	destroyAllWindows();

	// print them to user 
	std::cout << "There are "<< keypoints.size() << " SIFT feature found!" << std::endl;

	for (int i = 0; i < keypoints.size(); i++)
	{
		std::cout << std::endl << keypoints.at(i).pt << std::endl;
	}

	return keypoints;
}


// the method applies BoW algorithm
// Visualize the each test image's histogram
// input
// img: trainig images
// test_img: test images

void FeatureProc::siftBOW(vector<Mat> img, vector<Mat> test_img)
{
	// define vocabulary size
	int vocab_size = 500;

	// Create a trainer with vocab size 
	BOWKMeansTrainer bow_trainer(vocab_size);

	Mat descriptor;

	for(int i = 1; i < img.size(); i++)
	{
		// Extract keypoints
		SiftFeatureDetector detector;

		vector<KeyPoint> keypoints;

		detector.detect(img.at(i), keypoints);

		// Extract sift descriptors
		SiftDescriptorExtractor extractor;

		extractor.compute(img.at(i), keypoints, descriptor);

		// add descriptors to trainer
		bow_trainer.add(descriptor);
	}

	// create vocab
	Mat vocab = bow_trainer.cluster();

	// SHOW OUR vocabulary
	imshow("vocabulary", vocab);

	waitKey(0);

	// Cmputation of histogram 
	Ptr<FeatureDetector> features = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptors = DescriptorExtractor::create("SIFT");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

	BOWImgDescriptorExtractor bowDE(descriptors, matcher);

	std::cout << vocab.size();

	bowDE.setVocabulary(vocab);

	// TrainData
	Mat train;

	// labels
	vector<int> labels;

	// compute historam 
	for(int i = 0; i < test_img.size(); i++)
	{
		// Extract keypoints again
		SiftFeatureDetector detector;

		vector<KeyPoint> keypoints;

		detector.detect(img.at(i), keypoints);

		bowDE.compute(img.at(i), keypoints, descriptor);

		// in case of using SVM or sth I put them in a vector
		train.push_back(descriptor);

		// Store the label
		labels.push_back(i);

		// show the histogam of this image
		showHistogram(descriptor);

		// show the corersponding image
		imshow("image", test_img.at(i));

		waitKey(0);
	}

	destroyAllWindows();

	return;

}

// This function completely taken from https://opencv-code.com/tutorials/drawing-histograms-in-opencv/
// visualizing the histogram of visual words
void FeatureProc::showHistogram(Mat& img)
{
	int bins = 256;             // number of bins
	int nc = img.channels();    // number of channels
	vector<Mat> hist(nc);       // array for storing the histograms
	vector<Mat> canvas(nc);     // images for displaying the histogram
	int hmax[3] = {0,0,0};      // peak value for each histogram

	// The rest of the code will be placed here
	for (int i = 0; i < hist.size(); i++)
		hist[i] = Mat::zeros(1, bins, CV_32SC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < nc; k++)
			{
				uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<Vec3b>(i,j)[k];
				hist[k].at<int>(val) += 1;
			}
		}
	}
	for (int i = 0; i < nc; i++)
	{
		for (int j = 0; j < bins-1; j++)
			hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
	}
	const char* wname[3] = { "blue", "green", "red" };
	Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

	for (int i = 0; i < nc; i++)
	{
		canvas[i] = Mat::ones(125, bins, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < bins-1; j++)
		{
			line(
				canvas[i], 
				Point(j, rows), 
				Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])), 
				nc == 1 ? Scalar(200,200,200) : colors[i], 
				1, 8, 0
				);
		}

		imshow(nc == 1 ? "value" : wname[i], canvas[i]);
	}
}