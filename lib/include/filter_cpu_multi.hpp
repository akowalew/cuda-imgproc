///////////////////////////////////////////////////////////////////////////////
// filter_cpu_multi.hpp
//
// Contains declarations of image filtering functions
// Multi CPU implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 27.12.2019 16:24 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

void filter(const Image& src, Image& dst, const Image& kernel);
