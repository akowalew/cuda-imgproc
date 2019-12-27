///////////////////////////////////////////////////////////////////////////////
// hist_reference.cpp
//
// Contains definitions of functions working on images histograms
// Reference version (OpenCV based)
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 23:38 CEST
///////////////////////////////////////////////////////////////////////////////

#include "hist.hpp"

#include <opencv2/imgproc.hpp>

void equalize_hist(const Image& src, Image& dst)
{
	// Use directly OpenCV reference implementation
	equalizeHist(src, dst);
}