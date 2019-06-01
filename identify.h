#pragma once

#include <opencv2/core/mat.hpp>

float get_error(cv::Mat &temp, cv::Mat &obj);
void transform_image(cv::Mat &src, cv::Mat &des, cv::Point *points);
