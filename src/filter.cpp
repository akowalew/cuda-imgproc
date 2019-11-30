///////////////////////////////////////////////////////////////////////////////
// filter.cpp
//
// Contains definitions of functions related to image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////

#include "filter.hpp"
#include "timer.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

void filter2d_8_ref(const Image& src, Image& dst, const Image& kernel)
{
	const auto ddepth = -1; // Keep depth in destination same as in source
	cv::filter2D(src, dst, ddepth, kernel);
}

void filter2d_8_vserial(const Image& src, Image& dst, const Image& kernel)
{
    int K = kernel.rows, L = kernel.cols;
    for(int i=0; i<src.rows; ++i)
        for(int j=0; j<src.cols; ++j) {
            dst.at<unsigned char>(i, j) = 0;
            for(int k=0; k<kernel.rows; ++k) 
                for(int l=0; l<kernel.cols; ++l) {
                    if(j+l-L/2 >= 0 && i+k-K/2 >= 0 && j+l-L/2 < L && i+k-K/2 < K)
                        dst.at<unsigned char>(i, j) += kernel.at<unsigned char>(k, l) * src.at<unsigned char>(i+k-K/2, j+l-L/2);
                }
        }
}

void filter2d_8(const Image& src, Image& dst, const Image& kernel) {
    app_timer_t start, stop;
    timer(&start);
    filter2d_8_ref(src, dst, kernel);
    timer(&stop);
    elapsed_time(start, stop, 2 * src.cols * src.rows * kernel.cols * kernel.rows);
}
