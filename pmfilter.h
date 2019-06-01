#pragma once

#include <opencv2/opencv.hpp>

class perona_malik{
public:
    cv::Mat output;
    cv::Mat r[2], g[2], b[2], grad, c;
    int width;
    int height;
    bool mat_switch;

    perona_malik();

    void compute_grad(cv::Mat &img);
    void compute_cond(float lambda);
    void decompose(cv::Mat& img);
    void compose();
    void diffusion();
    cv::Mat& filter(cv::Mat& img, float lambda, int iterations);
};

