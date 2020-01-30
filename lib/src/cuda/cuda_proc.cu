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

static void cuda_process_host_image(CudaHostImage& h_dst, const CudaHostImage& h_src, const ProcessConfig& config)
{
	// Ensure correct images size
	assert(h_dst.cols == h_src.cols);
	assert(h_dst.rows == h_src.rows);

	const auto cols = h_src.cols;
	const auto rows = h_src.rows;
	const auto filter_ksize = config.filter_ksize;
	const auto median_ksize = config.median_ksize;

	LOG_INFO("Processing host image with CUDA\n");

	// Allocate buffers (synchronization point)
	auto d_kernel = cuda_create_kernel(filter_ksize);
	auto d_img_a = cuda_create_image(cols, rows);
	auto d_img_b = cuda_create_image(cols, rows);

	// Copy host data (synchronization point)
	cuda_image_copy_from_host(d_img_a, h_src);

	// Prepare kernel and set is as current for conv filtering (synchronization point)
	cuda_kernel_mean_blurr(d_kernel);
	cuda_set_filter_kernel_async(d_kernel);

	// Do right processing
	cuda_median_async(d_img_b, d_img_a, median_ksize);
	cuda_filter_async(d_img_a, d_img_b);
	cuda_equalize_hist_async(d_img_b, d_img_a);
	
	// Copy device data (synchronization point)
	cuda_image_copy_to_host(h_dst, d_img_b);

	// Free temporary buffers (synchronization point)
	cuda_free_kernel(d_kernel);
	cuda_free_image(d_img_a);
	cuda_free_image(d_img_b);
}

CudaHostImage cuda_process_host_image(const CudaHostImage& src, const ProcessConfig& config)
{
	auto dst = cuda_create_host_image(src.cols, src.rows);

	cuda_process_host_image(dst, src, config);

	return dst;
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

Image cuda_process_image(const Image& src, const ProcessConfig& config)
{
	return cuda_process_host_image(src, config);
}