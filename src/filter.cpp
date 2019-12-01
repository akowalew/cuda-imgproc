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

#include <opencv2/imgproc.hpp>
#include <stdio.h>

void filter2d_8(const Image& src, Image& dst, const Image& kernel) {
#ifdef DEBUG
    printf("Convolution filter call: ");
    app_timer_t start, stop;
    timer(&start);
#endif
    
    _filter2d_8(src, dst, kernel);
    
#ifdef DEBUG
    timer(&stop);
    elapsed_time(start, stop, 2 * src.cols * src.rows * kernel.cols * kernel.rows);
#endif
}
