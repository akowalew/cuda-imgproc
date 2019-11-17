///////////////////////////////////////////////////////////////////////////////
// histogram.hpp
//
// Contains declarations of functions working on images histograms
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 20:24 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace imgproc {

void equalize_hist_8(const unsigned char* src, unsigned char* dst, int length);

} // namespace imgproc
