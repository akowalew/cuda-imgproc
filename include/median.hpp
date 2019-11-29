///////////////////////////////////////////////////////////////////////////////
// median.hpp
//
// Contains declarations of functions related to median image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 28.11.2019 23:22 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

void median2d_8(const Image& src, Image& dst, int kernel_size);
