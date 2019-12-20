///////////////////////////////////////////////////////////////////////////////
// hist_types.hpp
//
// Contains declarations of types related to image histograms
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 20:02 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <limits>

//! Helper typedef - defines Histogram container for image with given depth
template<typename T>
using Histogram = std::array<
	int, //! Internal type of values counter
	(int)std::numeric_limits<T>::max() + 1 //! Number of elements - image depth
>;

//! Helper typedef - definition of Histogram for 8-bit images
using HistogramU8 = Histogram<unsigned char>;
