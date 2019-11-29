///////////////////////////////////////////////////////////////////////////////
// hist.cpp
//
// Contains definitions of functions working on images histograms
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 20:28 CEST
///////////////////////////////////////////////////////////////////////////////

#include "hist.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

void equalize_hist_8(const Image& src, Image& dst)
{
	cv::equalizeHist(src, dst);
}
