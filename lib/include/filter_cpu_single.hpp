///////////////////////////////////////////////////////////////////////////////
// filter_cpu_single.hpp
//
// Contains declarations of image filtering functions
// Single CPU implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 22.12.2019 16:12 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

void filter(const Image& src, Image& dst, const Image& kernel);
