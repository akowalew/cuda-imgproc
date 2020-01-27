///////////////////////////////////////////////////////////////////////////////
// cuda_proc.cu
//
// Contains definitions of functions related to CUDA image processor module
///////////////////////////////////////////////////////////////////////////////

#include "cuda_proc.cuh"

#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "cuda_filter.cuh"
#include "cuda_hist.cuh"
#include "cuda_image.cuh"
#include "cuda_kernel.cuh"
#include "cuda_median.cuh"
#include "log.hpp"

//
// Private functions
//

static void cuda_set_device(int device)
{
	LOG_INFO("Setting CUDA device no. %d\n", device);

    checkCudaErrors(cudaSetDevice(device));
}

static void cuda_reset_device()
{
	LOG_INFO("Resetting current CUDA device\n");

	checkCudaErrors(cudaDeviceReset());
}

static CudaHostImage cuda_process_host_image(const CudaHostImage& h_src, const ProcessConfig& config)
{
	LOG_INFO("Processing host image with CUDA\n");

	// Prepare buffers
	auto d_src = cuda_image_clone_from_host(h_src);
	auto d_kernel = cuda_create_mean_blurr_kernel(config.filter_ksize);

	// Do right processing
	auto d_medianed = cuda_median(d_src, config.median_ksize);
	auto d_filtered = cuda_filter(d_medianed, d_kernel);
	auto d_equalized = cuda_equalize_hist(d_filtered);
	
	// Copy results
	auto h_dst = cuda_image_clone_to_host(d_equalized);

	// Free temporary buffers
	cuda_free_image(d_equalized);
	cuda_free_image(d_filtered);
	cuda_free_image(d_medianed);
	cuda_free_kernel(d_kernel);
	cuda_free_image(d_src);

	return h_dst;
}

//
// Public functions
//

void cuda_proc_init()
{
	LOG_INFO("Initializing CUDA proc module\n");

	// Initialize device
	cuda_set_device(0);

	// Initialize modules
	cuda_hist_init();
}

void cuda_proc_deinit()
{
	LOG_INFO("Deinitializing CUDA proc module\n");

	// Deinitialize modules
	cuda_hist_deinit();

	// Deinitialize device
	cuda_reset_device();
}

Image cuda_process_image(const Image& img, const ProcessConfig& config)
{
	return cuda_process_host_image(img, config);
}
