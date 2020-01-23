///////////////////////////////////////////////////////////////////////////////
// kernel.hpp
//
// Contains declarations of functions related to image filters kernels
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "types.hpp"

struct Kernel;

using KernelSize = size_t;

using KernelValue = float;

Kernel* make_kernel(KernelSize size);

Kernel* make_kernel(KernelSize size, KernelValue value);

Kernel* mean_blurr_kernel(KernelSize size);

Kernel* gaussian_kernel(KernelSize size);

void free_kernel(Kernel* kernel);

void set_kernel_data(Kernel* kernel, const void* data, size_t size);

void get_kernel_data(Kernel* kernel, void* data, size_t size);

