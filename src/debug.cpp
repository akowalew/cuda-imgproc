///////////////////////////////////////////////////////////////////////////////
// debug.cpp
//
// Contains definitions of image debug functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 28.11.2019 1:18 CEST
///////////////////////////////////////////////////////////////////////////////

#include "debug.hpp"

#include <cassert>

#include <opencv2/highgui.hpp>

void show_image(const char* name, Image image)
{
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, image);
}

void wait_for_exit()
{
	cv::waitKey(0);
}
