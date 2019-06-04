#pragma once

#include <opencv2/core/mat.hpp>

float get_error(cv::Mat &temp, cv::Mat &obj);
void identify_my_transform_image(cv::Mat &src, cv::Mat &des, cv::Point *points);
