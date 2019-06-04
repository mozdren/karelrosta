#include "pmfilter.h"

perona_malik::perona_malik(){
    this->width = -1;
    this->height = -1;
    this->mat_switch = true;
}

void perona_malik::compute_grad(cv::Mat &img){
    float dxr,dyr,dxg,dyg,dxb,dyb,dx,dy;
    for (auto y = 0; y < this->height; y++){
        for (auto x = 0; x < this->width; x++){
            if (x == 0){
                auto other = img.at<cv::Vec3b>(y, x + 1);
                auto center = img.at<cv::Vec3b>(y, x);
                dxr = std::abs(static_cast<float>(other[0]) - static_cast<float>(center[0]));
                dxg = std::abs(static_cast<float>(other[1]) - static_cast<float>(center[1]));
                dxb = std::abs(static_cast<float>(other[2]) - static_cast<float>(center[2]));
                dx = std::max(dxr, std::max(dxg, dxb));
            }else if(x == img.cols-1){
                auto other = img.at<cv::Vec3b>(y, x - 1);
                auto center = img.at<cv::Vec3b>(y, x);
                dxr = std::abs(static_cast<float>(other[0]) - static_cast<float>(center[0]));
                dxg = std::abs(static_cast<float>(other[1]) - static_cast<float>(center[1]));
                dxb = std::abs(static_cast<float>(other[2]) - static_cast<float>(center[2]));
                dx = std::max(dxr, std::max(dxg, dxb));
            }else{
                auto other = img.at<cv::Vec3b>(y, x - 1);
                auto center = img.at<cv::Vec3b>(y, x + 1);
                dxr = std::abs(static_cast<float>(other[0]) - static_cast<float>(center[0]));
                dxg = std::abs(static_cast<float>(other[1]) - static_cast<float>(center[1]));
                dxb = std::abs(static_cast<float>(other[2]) - static_cast<float>(center[2]));
                dx = std::max(dxr, std::max(dxg, dxb));
            }

            if (y == 0){
                auto other = img.at<cv::Vec3b>(y + 1, x);
                auto center = img.at<cv::Vec3b>(y, x);
                dxr = std::abs(static_cast<float>(other[0]) - static_cast<float>(center[0]));
                dxg = std::abs(static_cast<float>(other[1]) - static_cast<float>(center[1]));
                dxb = std::abs(static_cast<float>(other[2]) - static_cast<float>(center[2]));
                dy = std::max(dyr, std::max(dyg, dyb));
            }else if(y == img.rows-1){
                auto other = img.at<cv::Vec3b>(y - 1, x);
                auto center = img.at<cv::Vec3b>(y, x);
                dxr = std::abs(static_cast<float>(other[0]) - static_cast<float>(center[0]));
                dxg = std::abs(static_cast<float>(other[1]) - static_cast<float>(center[1]));
                dxb = std::abs(static_cast<float>(other[2]) - static_cast<float>(center[2]));
                dy = std::max(dyr, std::max(dyg,dyb));
            }else{
                auto other = img.at<cv::Vec3b>(y + 1, x);
                auto center = img.at<cv::Vec3b>(y - 1, x);
                dxr = std::abs(static_cast<float>(other[0]) - static_cast<float>(center[0]));
                dxg = std::abs(static_cast<float>(other[1]) - static_cast<float>(center[1]));
                dxb = std::abs(static_cast<float>(other[2]) - static_cast<float>(center[2]));
                dy = std::max(dyr, std::max(dyg,dyb));
            }

            this->grad.at<float>(y, x) = std::max(dx,dy);
        }
    }
    //GaussianBlur(this->grad,this->grad, cv::Size(3, 3), 0);
}

void perona_malik::compute_cond(const float lambda){
    for (auto y = 0; y < this->height; y++){
        for (auto x = 0; x < this->width; x++){
            auto g = this->grad.at<float>(y, x);
            this->c.at<float>(y, x) = 1.0f/(1.0f + g / lambda * (g / lambda));
        }
    }
}

void perona_malik::decompose(cv::Mat &img){
    int i;
    if (this->mat_switch) i = 0;
    else i = 1;
    for (auto y = 0;y<this->height;y++){
        for (auto x = 0;x<this->width;x++){
            auto pixel = img.at<cv::Vec3b>(y, x);
            this->r[i].at<float>(y, x) = static_cast<float>(pixel[0]);
            this->g[i].at<float>(y, x) = static_cast<float>(pixel[1]);
            this->b[i].at<float>(y, x) = static_cast<float>(pixel[2]);
        }
    }
}

void perona_malik::compose(){
    int i;
    if (this->mat_switch) i = 0;
    else i = 1;
    for (auto y = 0; y<this->height; y++){
        for (auto x = 0; x<this->width; x++){
            cv::Vec3b pixel;
            pixel[0] = static_cast<unsigned>(this->r[i].at<float>(y, x));
            pixel[1] = static_cast<unsigned>(this->g[i].at<float>(y, x));
            pixel[2] = static_cast<unsigned>(this->b[i].at<float>(y, x));
            this->output.at<cv::Vec3b>(y, x) = pixel;
        }
    }
}

void perona_malik::diffusion(){
    int i,i2;
    const auto alpha = 0.1f;

    if (this->mat_switch){
        i=0;
        i2=1;
    }else{
        i=1;
        i2=0;
    }

    for (auto y = 0; y < this->height; y++){
        for (auto x = 0; x < this->width; x++){
            auto sum_r = this->r[i].at<float>(y, x);
            auto sum_g = this->g[i].at<float>(y, x);
            auto sum_b = this->b[i].at<float>(y, x);
            // for RED
            if (x != 0) sum_r += alpha*this->c.at<float>(y,x-1)*(this->r[i].at<float>(y,x-1) - this->r[i].at<float>(y,x));
            if (x != this->width-1) sum_r += alpha*this->c.at<float>(y,x+1)*(this->r[i].at<float>(y,x+1) - this->r[i].at<float>(y,x));
            if (y != 0) sum_r += alpha*this->c.at<float>(y-1,x)*(this->r[i].at<float>(y-1,x) - this->r[i].at<float>(y,x));
            if (y != this->height-1) sum_r += alpha*this->c.at<float>(y+1,x)*(this->r[i].at<float>(y+1,x) - this->r[i].at<float>(y,x));
            // for GREEN
            if (x != 0) sum_g += alpha*this->c.at<float>(y,x-1)*(this->g[i].at<float>(y,x-1) - this->g[i].at<float>(y,x));
            if (x != this->width-1) sum_g += alpha*this->c.at<float>(y,x+1)*(this->g[i].at<float>(y,x+1) - this->g[i].at<float>(y,x));
            if (y != 0) sum_g += alpha*this->c.at<float>(y-1,x)*(this->g[i].at<float>(y-1,x) - this->g[i].at<float>(y,x));
            if (y != this->height-1) sum_g += alpha*this->c.at<float>(y+1,x)*(this->g[i].at<float>(y+1,x) - this->g[i].at<float>(y,x));
            // for BLUE
            if (x != 0) sum_b += alpha*this->c.at<float>(y,x-1)*(this->b[i].at<float>(y,x-1) - this->b[i].at<float>(y,x));
            if (x != this->width-1) sum_b += alpha*this->c.at<float>(y,x+1)*(this->b[i].at<float>(y,x+1) - this->b[i].at<float>(y,x));
            if (y != 0) sum_b += alpha*this->c.at<float>(y-1,x)*(this->b[i].at<float>(y-1,x) - this->b[i].at<float>(y,x));
            if (y != this->height-1) sum_b += alpha*this->c.at<float>(y+1,x)*(this->b[i].at<float>(y+1,x) - this->b[i].at<float>(y,x));
            this->r[i2].at<float>(y,x) = sum_r;
            this->g[i2].at<float>(y,x) = sum_g;
            this->b[i2].at<float>(y,x) = sum_b;
        }
    }
    this->mat_switch = !this->mat_switch;
}

cv::Mat & perona_malik::filter(cv::Mat & img, const float lambda, const int iterations){
    if (this->width == -1){
        this->width = img.cols;
        this->height = img.rows;
        this->output = cv::Mat(img.rows, img.cols, CV_8UC3);
        this->r[0] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->r[1] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->g[0] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->g[1] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->b[0] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->b[1] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->grad = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->c = cv::Mat(img.rows, img.cols, CV_32FC1);
    }
    if (img.cols != output.cols || img.rows != output.rows){
        this->width = img.cols;
        this->height = img.rows;
        this->output = cv::Mat(img.rows, img.cols, CV_8UC3);
        this->r[0] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->r[1] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->g[0] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->g[1] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->b[0] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->b[1] = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->grad = cv::Mat(img.rows, img.cols, CV_32FC1);
        this->c = cv::Mat(img.rows, img.cols, CV_32FC1);
    }
    this->compute_grad(img);
    this->compute_cond(lambda);
    this->decompose(img);
    for (auto i = 0;i < iterations; i++){
        this->diffusion();
    }
    this->compose();
    return output;
}

