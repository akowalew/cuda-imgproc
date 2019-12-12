///////////////////////////////////////////////////////////////////////////////
// reader.cpp
//
// Contains definitions of functions related to image reading
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 29.11.2019 0:47 CEST
///////////////////////////////////////////////////////////////////////////////


#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include <cassert>
#include "reader.hpp"


Image read_image(const char* path)
{
#if (CV_MAJOR_VERSION < 3)
    auto image = cv::imread(path, CV_LOAD_IMAGE_COLOR);
#else
    auto image = cv::imread(path, cv::IMREAD_COLOR);
#endif
    assert(image.type() == CV_8UC3);
    assert(image.channels() == 3);
    return image;
}