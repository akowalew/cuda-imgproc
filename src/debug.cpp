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

namespace {

std::vector<Image> g_images; //! Global array of images to be displayed

} //

void show_image(const char* name, Image image)
{
	// First, clone that image to have local copy
	auto it = g_images.insert(g_images.end(), image.clone());
	auto& cloned = *it;

	// Then, display our cloned image
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, cloned);
}

void wait_for_exit()
{
	cv::waitKey(0);
}
