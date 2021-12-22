//opencv libraries
//#include "opencv2/core/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"



//C++ standard libraries
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

//Global variables
int threshval = 128;
int max_thresh = 255;



//function declarations

/**
Function that returns the maximum of 3 integers
@param a first integer
@param b second integer
@param c third integer
*/
int myMax(int a, int b, int c);

/**
Function that returns the minimum of 3 integers
@param a first integer
@param b second integer
@param c third integer
*/
int myMin(int a, int b, int c);

/**
Function that detects whether a pixel belongs to the skin based on RGB values
@param src The source color image
@param dst The destination grayscale image where skin pixels are colored white and the rest are colored black
*/
void mySkinDetect(Mat& src, Mat& dst);



int match_method; int max_Trackbar = 5;

Mat temp_up; Mat temp_up_gray; Mat src_img; Mat temp_img;

const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";

int main()
{

	//namedWindow("binary up", WINDOW_AUTOSIZE);
	namedWindow("post templatematch", WINDOW_AUTOSIZE);
	namedWindow("template", WINDOW_AUTOSIZE);
	//namedWindow("skin", WINDOW_AUTOSIZE);

	//read template: thumbs up, in grayscale 
		//name is temp_up because that's the first template i was testing 
	
	
	Mat temp_up = imread("thumb-temp1.jpg", 0); //thumbs up template 

	//Here are the other templates commented out to test them: 

	 //Mat temp_up = imread("palm-temp0.jpg", 0);  //open palm template
	 //Mat temp_up = imread("down-temp1.jpg", 0);  // thumbs down template
	//Mat temp_up = imread("peace-temp0.jpg", 0); // peace sign template 

	if (!temp_up.data) { cout << "File not found" << std::endl; return -1; };


	int erosion_size = 3;
	int dilation_size = 3;
	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));

	//perform erosions and dilations
	erode(temp_up, temp_up, element);
	dilate(temp_up, temp_up, element);

	threshold(temp_up, temp_up_gray, 100, 255, THRESH_BINARY);


	Mat bin_temp_up;
	bin_temp_up = Mat::zeros(temp_up_gray.size(), CV_8UC1);

	threshold(temp_up_gray, bin_temp_up, 100, 255, THRESH_BINARY);

	dilate(bin_temp_up, bin_temp_up, element);

	//show the binary image of the template 
	imshow("template", bin_temp_up); 

	//Read the source image
	Mat src_img = imread("thumb-test1.jpg", 1); //thumbs up test image

	//Other source images:
	//Mat src_img = imread("palm-test0.jpg", 1); //palm test image
	//Mat src_img = imread("down-test1.jpg", 1); //thumbs down test image
	//Mat src_img = imread("peace-test0.jpg", 1); //peace test image


	if (!src_img.data) { cout << "File not found" << std::endl; return -1; };

	Mat dest_img;
	dest_img = Mat::zeros(src_img.size(), CV_8UC1);

	Mat bin_dest_img;
	bin_dest_img = Mat::zeros(src_img.size(), CV_8UC1);

	//perform skin detection on the source image
	mySkinDetect(src_img, bin_dest_img);

	//template matching keeps reading my wrist or face as the match, so i'm dilating the source image to make it more accurate.
	dilate(bin_dest_img, bin_dest_img, element);
	dilate(bin_dest_img, bin_dest_img, element);

	//imshow("skin", bin_dest_img);

	//template matching -- taken from openCV forums
	Mat src_display; Mat result;
	src_img.copyTo(src_display);

	int result_cols = src_img.cols - temp_up.cols + 1;
	int result_rows = src_img.rows - temp_up.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	matchTemplate(bin_dest_img, bin_temp_up, result, CV_TM_SQDIFF);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	matchLoc = minLoc;

	rectangle(src_display, matchLoc, Point(matchLoc.x + temp_up.cols, matchLoc.y + temp_up.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + temp_up.cols, matchLoc.y + temp_up.rows), Scalar::all(0), 2, 8, 0);

	//show final result
	imshow("post templatematch", src_display);



	waitKey(0);
	return 0;

}

//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b)); //short-circuit evaluation
	(void)((m < c) && (m = c));
	return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}


//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			//For each pixel, compute the average intensity of the 3 color channels
			Vec3b intensity = src.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			//LAB#3 Comment: change values below to better match your own skin tone 
			if ((R > 90 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B)) {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}


