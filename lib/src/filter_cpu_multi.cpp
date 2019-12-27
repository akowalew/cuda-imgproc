#include "filter_cpu_multi.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

void filter(const Image& src, Image& dst, const Image& kernel)
{
    int K = kernel.rows, L = kernel.cols;
    for(int i=0; i<src.rows; ++i)
        for(int j=0; j<src.cols; ++j) {;
            float acc = 0.0f;
            for(int k=0; k<kernel.rows; ++k)
                for(int l=0; l<kernel.cols; ++l) {
                    if(j+l-L/2 >= 0 && i+k-K/2 >= 0 && j+l-L/2 < src.cols && i+k-K/2 < src.rows) {
                        acc += kernel.at<float>(k, l) * src.at<unsigned char>(i+k-K/2, j+l-L/2);
                    }
                }

            acc = round(acc);
            if(acc < 0.0f)
                acc = 0.0f;
            if(acc > 255.0f)
                acc = 255.0f;
            dst.at<unsigned char>(i, j) = acc;
        }
}

