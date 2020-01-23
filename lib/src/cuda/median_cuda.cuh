///////////////////////////////////////////////////////////////////////////////
// median.hpp
//
// Contains declarations of functions related to median image filtering
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "types.hpp"

//
// Forward declarations
//

struct Image;

//
// Public declarations
//

using MedianKernelSize = size_t;

void median(Image* dst, const Image* src, MedianKernelSize ksize);

Image* median(const Image* src, MedianKernelSize ksize);