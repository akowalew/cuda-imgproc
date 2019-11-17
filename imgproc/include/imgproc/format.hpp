///////////////////////////////////////////////////////////////////////////////
// format.hpp
//
// Contains declarations of functions related to image format conversions.
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 17:17 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace imgproc {

void bgr888_to_hsv888(const unsigned char* bgr, unsigned char* hsv, int rows, int ncols);

} // namespace imgproc
