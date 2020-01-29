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
// Private functions
//

static Image cpu_process_host_image(const Image& src, const ProcessConfig& config)
{
	LOG_INFO("Processing host image with CUDA\n");

	// Prepare buffers
    const auto filter_ksize = config.filter_ksize;
    const auto kernel_v = (1.0f / (filter_ksize * filter_ksize));
    auto kernel = cv::Mat_<float>(filter_ksize, filter_ksize, kernel_v);

	// Do right processing
	auto medianed = cpu_median(src, config.median_ksize);
	auto filtered = cpu_filter(medianed, kernel);
	auto equalized = cpu_equalize_hist(filtered);
	
	return equalized;
}

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

Image cpu_process_image(const Image& img, const ProcessConfig& config)
{
	return cpu_process_host_image(img, config);
}
