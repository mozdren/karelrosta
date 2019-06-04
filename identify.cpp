#include "identify.h"
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

float get_error(cv::Mat &temp, cv::Mat &obj){
    auto sum = 0.0f;
    auto count = 0.0f;
	for (auto y = 0;y<temp.rows; y++){
		for (auto x = 0;x<temp.cols; x++){
		    const auto diff = static_cast<int>(temp.at<uchar>(y, x)) - static_cast<int>(obj.at<uchar>(y, x));
			sum += static_cast<float>(diff * diff);
			count += 1.0f;
		}
	}
	return sum/count;
}

void identify_my_transform_image(cv::Mat &src, cv::Mat &des, cv::Point *points){
    cv::Mat input_mat = cv::Mat(4, 2, CV_32FC1);
    cv::Mat output_mat = cv::Mat(4, 2, CV_32FC1);
    cv::Mat H = cv::Mat(3, 3, CV_32FC1);
    
    output_mat.at<float>(0, 0) = 0.0f;
    output_mat.at<float>(0, 1) = 0.0f;
    output_mat.at<float>(1, 0) = static_cast<float>(des.cols);
    output_mat.at<float>(1, 1) = 0.0f;
    output_mat.at<float>(2, 0) = static_cast<float>(des.cols);
    output_mat.at<float>(2, 1) = static_cast<float>(des.rows);
    output_mat.at<float>(3, 0) = 0.0f;
    output_mat.at<float>(3, 1) = static_cast<float>(des.rows);

    input_mat.at<float>(0, 0) = static_cast<float>(points[0].x);
    input_mat.at<float>(0, 1) = static_cast<float>(points[0].y);
    input_mat.at<float>(1, 0) = static_cast<float>(points[1].x);
    input_mat.at<float>(1, 1) = static_cast<float>(points[1].y);
    input_mat.at<float>(2, 0) = static_cast<float>(points[2].x);
    input_mat.at<float>(2, 1) = static_cast<float>(points[2].y);
    input_mat.at<float>(3, 0) = static_cast<float>(points[3].x);
    input_mat.at<float>(3, 1) = static_cast<float>(points[3].y);

    findHomography(input_mat, output_mat, H);
    warpPerspective(src, des, H, cv::Size(des.cols, des.rows));
}


