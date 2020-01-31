///////////////////////////////////////////////////////////////////////////////
// cpu_proc.cpp
//
// Contains definitions of functions related to CUDA image processor module
///////////////////////////////////////////////////////////////////////////////

#include "cpu_proc.hpp"

#include <cstdio>

#include "cpu_filter.hpp"
#include "cpu_hist.hpp"
#include "cpu_median.hpp"
#include "log.hpp"

//
// Public functions
//

void cpu_proc_init()
{
	LOG_INFO("Initializing CPU proc module\n");
}

void cpu_proc_deinit()
{
	LOG_INFO("Deinitializing CPU proc module\n");
}

Image cpu_process_image(const Image& src, const Kernel& kernel, size_t median_ksize)
{
	LOG_INFO("Processing host image with CUDA\n");

	// Do right processing
	auto medianed = cpu_median(src, median_ksize);
	auto filtered = cpu_filter(medianed, kernel);
	auto equalized = cpu_equalize_hist(filtered);
	
	return equalized;
}