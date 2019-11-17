///////////////////////////////////////////////////////////////////////////////
// filter.hpp
//
// Contains declarations of functions related to image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 17:19 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace imgproc {

void filter2d_888(const unsigned char* src, unsigned char* dst,
	int ncols, int nrows,
	const float* kernel, int kernel_size);

} // namespace imgproc
