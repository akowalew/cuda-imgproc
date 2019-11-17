///////////////////////////////////////////////////////////////////////////////
// core.hpp
//
// Contains declarations of core image processing functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 17:19 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace imgproc {

void split_888(int nrows, int ncols,
	const unsigned char* src,
	unsigned char* x, unsigned char* y, unsigned char* z);

void merge_888(int nrows, int ncols,
	const unsigned char* src_x, const unsigned char* src_y, const unsigned char* src_z,
	unsigned char* dst);

} // namespace imgproc
