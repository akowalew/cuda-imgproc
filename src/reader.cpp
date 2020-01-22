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
#include<string>

using namespace std;

Image read_image(const char* path)
{
#if (CV_MAJOR_VERSION < 3)
    auto image = cv::imread(path, CV_LOAD_IMAGE_COLOR);
#else
    auto image = cv::imread(path, cv::IMREAD_COLOR);
#endif

	if (1) {
		string r;

		uchar depth = image.type() & CV_MAT_DEPTH_MASK;
		uchar chans = 1 + (image.type() >> CV_CN_SHIFT);

		switch (depth) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
		}
		r += "C";
		r += (chans + '0');
		printf("Matrix: %s %dx%d \n", r.c_str(), image.cols, image.rows);
	}

    assert(image.type() == CV_8UC3);
    assert(image.channels() == 3);
    return image;
}