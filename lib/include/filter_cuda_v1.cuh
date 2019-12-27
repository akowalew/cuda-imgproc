///////////////////////////////////////////////////////////////////////////////
// filter_cuda_v1.cuh
//
// Contains declarations of image filtering functions
// CUDA v1 implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 27.12.2019 21:17 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

void filter(const Image& src, Image& dst, const Image& kernel);
