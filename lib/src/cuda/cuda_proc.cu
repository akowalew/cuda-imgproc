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

static void cuda_process_image_async(
	CudaImage& img_b, CudaImage& img_a, 
	size_t filter_ksize, size_t median_ksize)
{
	// Do right processing asynchronously
	cuda_median_async(img_b, img_a, median_ksize);
	cuda_filter_async(img_a, img_b, filter_ksize);
	cuda_equalize_hist_async(img_b, img_a);
}

static void cuda_process_host_image_async(
	Image& h_dst, const Image& h_src, 
	const Kernel& h_filter_kernel, size_t median_ksize,
	CudaImage& d_img_a, CudaImage& d_img_b)
{
	// Copy data from host asynchronously
	cuda_image_copy_from_host_async(d_img_a, h_src);
	cuda_filter_copy_kernel_from_host_async(h_filter_kernel);

	// Perform image processing asynchronously
	const auto filter_ksize = h_filter_kernel.cols;
	cuda_process_image_async(d_img_b, d_img_a, filter_ksize, median_ksize);
		
	// Copy data to host asynchronously
	cuda_image_copy_to_host_async(h_dst, d_img_b);
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


Image cuda_process_host_image(
	const Image& h_src, 
	const Kernel& kernel, size_t median_ksize)
{
	const auto cols = h_src.cols;
	const auto rows = h_src.rows;

	LOG_INFO("Processing image with CUDA\n");
	
	// Allocate temporary host buffer
	auto h_dst = cuda_create_host_image(cols, rows);

	// Allocate temporary CUDA buffers (synchronization point)
	auto d_img_a = cuda_create_image(cols, rows);
	auto d_img_b = cuda_create_image(cols, rows);

	// Perform processing of host image asynchronously
	cuda_process_host_image_async(h_dst, h_src, kernel, median_ksize, d_img_a, d_img_b);

	// Free temporary CUDA buffers (synchronization point)
	cuda_free_image(d_img_a);
	cuda_free_image(d_img_b);

	return h_dst;
}