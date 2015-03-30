#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>

#include "lib_features.h"
#include "ib_color.h"


using namespace std; 
using namespace cv;

vector<Mat> readTraining();
vector<Mat> readTest();


int main(int argc, char** argv)
{
	// Variable Definitons

	// Create a instance of FeatureProc class and Colorful class
	FeatureProc featureProc;
	Colorful colorful;

	Mat harris_1, harris_2, img_1, img_2;

	vector<Mat> img_1_hsv, img_2_hsv;

	vector<Point> harris_points_vec_1, harris_points_vec_2;
	vector<KeyPoint> sift_keypoints_vec_1, sift_keypoints_vec_2;

	// variable that stores user input
	int user_input = 0; 

	// Flag to used in main loop
	int is_user_selected = 1;

	// Check the arguments
	if (argc < 3)
	{
		cout << "Not enough parameters!!" << endl;

		//return -1;
	}

	// Read the images
	img_1 = imread(argv[1]);
	img_2 = imread(argv[2]);

	// Get the intensity value of images
	img_1_hsv = colorful.convertBGR2HSV(img_1);
	img_2_hsv = colorful.convertBGR2HSV(img_2);

	// For Harris
	const Mat img_I_1_h = img_1_hsv.at(2);
	const Mat img_I_2_h = img_2_hsv.at(2);

	// For SIFT
	const Mat img_I_1_s = img_1_hsv.at(2);
	const Mat img_I_2_s = img_2_hsv.at(2);

	while (is_user_selected == 1)
	{
		// User Prompt 
		cout << "\nWhich one do you want?\n1 - Harris\n2 - SIFT\n3 - Bag of Words\n4 - Exit" << endl ;
		cin >> user_input;

		// route the user according to his/her selection
		if (user_input == 1)
		{
			int threshold;

			cout << "Enter the threshold: ";
			cin >> threshold;

			// user selected Harris Corner Detection
			harris_points_vec_1 = featureProc.harrisDetector(img_I_1_h, threshold);
			harris_points_vec_2 = featureProc.harrisDetector(img_I_2_h, threshold);

			// Set the flag 
			is_user_selected = 1;
		}
		else if(user_input == 2)
		{
			// user selected SIFT Detection

			sift_keypoints_vec_1 = featureProc.siftFind(img_I_1_s);
			sift_keypoints_vec_2 = featureProc.siftFind(img_I_2_s);

			// Set the flag 
			is_user_selected = 1;
		}
		else if (user_input == 3)
		{
			// user selected the Bag of Words 

			// read training and test data 
			vector<Mat> training = readTraining();
			vector<Mat> test = readTest();

			// send them to BoW function
			featureProc.siftBOW(training, test);

			// Set the flag
			is_user_selected = 1;
		}
		else 
		{
			// exit from the loop

			// Reset the flag
			is_user_selected = 0;
		}
	}

	return 0;
}

// method that reads all training images 
// for BoW section
// the training image size is harcoded as 20
// training images must be in training_images folder
// and labeled as train_1, train_2 ...
vector<Mat> readTraining()
{
	vector<Mat> training;

	for(int i = 1; i <= 20; i++)
	{
		// name of file
		std::stringstream sstm;
        sstm << "train_" << i << ".jpg";
        string name = sstm.str();

		// load them in grayscale
		Mat temp = imread(name);

		training.push_back(temp);

		imshow("training_images", temp);

		waitKey(300);
	}

	destroyAllWindows();

	return training;
}

// method that reads all test images 
// for BoW section
// the test image size is harcoded as 4
// test images must be in test_images folder
// and labeled as test_1, test_2 ...
vector<Mat> readTest()
{
	vector<Mat> test;

	for(int i = 1; i <= 4; i++)
	{
		// name of file
		std::stringstream sstm;
        sstm << "test_" << i << ".jpg";
        string name = sstm.str();

		// load them in grayscale
		Mat temp = imread(name);

		test.push_back(temp);

		imshow("test_images", temp);

		waitKey(300);
	}

	destroyAllWindows();

	return test;
}