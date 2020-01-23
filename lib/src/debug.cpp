///////////////////////////////////////////////////////////////////////////////
// debug.cu
//
// Contains definitions of image debug functions
///////////////////////////////////////////////////////////////////////////////

#include "debug.hpp"

#include <cassert>

#include <opencv2/highgui.hpp>

void show_image(Image image, const char* name)
{
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, image);
}

void wait_for_exit()
{
	cv::waitKey(0);
}
