///////////////////////////////////////////////////////////////////////////////
// format.cpp
//
// Contains implementation of functions related to image format conversions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:57 CEST
///////////////////////////////////////////////////////////////////////////////

#include "imgproc/format.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

namespace imgproc {

void bgr888_to_hsv888(const unsigned char* bgr, unsigned char* hsv, int cols, int rows)
{
    assert(hsv != nullptr);
    assert(bgr != nullptr);
    assert(cols > 0);
    assert(rows > 0);

    const auto type = CV_8UC3;
    const auto bgr_mat = cv::Mat(rows, cols, type, const_cast<unsigned char*>(bgr));
    auto hsv_mat = cv::Mat(rows, cols, type, hsv);

    cv::cvtColor(bgr_mat, hsv_mat, CV_BGR2HSV);
}

} // namespace imgproc
