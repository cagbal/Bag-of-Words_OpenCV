
#include <opencv2\core\core.hpp> // include core.hpp to use MAT
#include <vector>

/*
* class that includes some color methods          
*  
*                                                       
* written by Cagatay Odabasi                            
*
*/

// The name is to prevent from confusion
class Colorful 
{
public:
	Colorful(){}; // void constructor

	// method to extract BGR channels from a picture
	std::vector<cv::Mat> extractRGB(cv::Mat); 

	// method to convert BGR to HSV
	// and returns the each HSV channel seperately
	std::vector<cv::Mat> convertBGR2HSV(cv::Mat);

	// method to convert HSV to BGR
	// and returns the each BGR channel seperately
	std::vector<cv::Mat> convertHSV2BGR(cv::Mat);
};