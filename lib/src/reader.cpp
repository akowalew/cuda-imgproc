///////////////////////////////////////////////////////////////////////////////
// reader.cpp
// 
// Contains implementation of image reader module
///////////////////////////////////////////////////////////////////////////////

#include "reader.hpp"

#include <cstdio>
#include <cassert>

#include <opencv2/highgui.hpp>

Image read_image(const char* path)
{
    printf("*** Reading image from '%s'\n", path);

#if (CV_MAJOR_VERSION < 3)
    auto image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
#else
    auto image = cv::imread(path, cv::IMREAD_GRAYSCALE);
#endif
    assert(image.type() == CV_8UC1);
    assert(image.channels() == 1);

    return image;
}