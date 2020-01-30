///////////////////////////////////////////////////////////////////////////////
// kernel.hpp
//
// Contains declarations for Kernel manager
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2/core.hpp>

using Kernel = cv::Mat_<float>;

Kernel create_kernel(size_t size);

void free_kernel(Kernel& kernel);

Kernel create_mean_blurr_kernel(size_t ksize);