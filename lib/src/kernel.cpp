///////////////////////////////////////////////////////////////////////////////
// kernel.cpp
//
// Contains declarations for kernel manager
///////////////////////////////////////////////////////////////////////////////

#include "kernel.hpp"

#include "log.hpp"

Kernel create_kernel(size_t size)
{
    return Kernel(size, size);
}

void free_kernel(Kernel& kernel)
{
    kernel.release();
}

Kernel create_mean_blurr_kernel(size_t ksize)
{
    const auto ksize_sq = (ksize * ksize);
    const auto kernel_v = (1.0f / ksize_sq);
    return Kernel(ksize, ksize, kernel_v);
}