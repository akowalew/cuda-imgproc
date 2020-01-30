///////////////////////////////////////////////////////////////////////////////
// proc.cpp
//
// Contains definitions of functions related to image processor module
///////////////////////////////////////////////////////////////////////////////

#include "proc.hpp"

#include <cstdio>

#include "log.hpp"

#ifdef BUILD_VERSION_CUDA
# include "cuda_proc.cuh"
#else
# include "cpu_proc.hpp"
#endif

//
// Public functions
//

void proc_init()
{
	LOG_INFO("Initializing proc module\n");

#ifdef BUILD_VERSION_CUDA
	cuda_proc_init();
#else
	cpu_proc_init();
#endif
}

void proc_deinit()
{
	LOG_INFO("Deinitializing proc module\n");

#ifdef BUILD_VERSION_CUDA
	cuda_proc_deinit();
#else 
	cpu_proc_deinit();
#endif
}

Image process_image(const Image& img, const Kernel& filter_kernel, size_t median_ksize)
{
	LOG_INFO("Processing image\n");

#ifdef BUILD_VERSION_CUDA
	return cuda_process_host_image(img, filter_kernel, median_ksize);
#else
	return cpu_process_image(img, filter_kernel, median_ksize);
#endif
}
