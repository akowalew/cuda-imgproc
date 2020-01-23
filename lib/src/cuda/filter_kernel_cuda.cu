///////////////////////////////////////////////////////////////////////////////
// filter_kernel.cu
//
// Contains definitions of functions related to image filters kernels
///////////////////////////////////////////////////////////////////////////////

#include "filter_kernel.hpp"

Image mean_blurr_kernel(size_t size)
{
    // Create square-sized single-channel kernel of type float
    const auto rows = size;
    const auto cols = size;
    const auto type = CV_32F;
    auto kernel = Image{rows, cols, type};

    // Fill kernel with mean-blurr values
    const auto size_sq = (size * size);
    const auto kernel_begin = (float*)kernel.data;
    const auto kernel_end = (float*)kernel.dataend;
    const auto kernel_value = (1.0f / size_sq);
    std::fill(kernel_begin, kernel_end, kernel_value);

    return kernel;
}
