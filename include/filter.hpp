///////////////////////////////////////////////////////////////////////////////
// filter.hpp
//
// Contains declarations of functions related to image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 17:19 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

void filter2d_8(const Image& src, Image& dst, const Image& kernel);
void _filter2d_8(const Image& src, Image& dst, const Image& kernel);
