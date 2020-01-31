///////////////////////////////////////////////////////////////////////////////
// cuda_proc.cu
//
// Contains definitions of functions related to CUDA image processor module
///////////////////////////////////////////////////////////////////////////////

#include "cuda_proc.cuh"

#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "cuda_common.cuh"
#include "cuda_filter.cuh"
#include "cuda_hist.cuh"
#include "cuda_image.cuh"
#include "cuda_kernel.cuh"
#include "cuda_median.cuh"
#include "log.hpp"

//
// Private members
//

CudaImage g_img_a;
CudaImage g_img_b;

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

static void cuda_process_host_image_async(
	Image& h_dst, const Image& h_src, 
	const Kernel& h_filter_kernel, size_t median_ksize)
{
	assert(h_dst.cols == h_src.cols);
	assert(h_dst.rows == h_src.rows);

	const auto cols = h_src.cols;
	const auto rows = h_src.rows;
	const auto filter_ksize = h_filter_kernel.cols;

	// Form subimages from local images, suited to given size
	auto img_a = cuda_image_sub(g_img_a, cols, rows);
	auto img_b = cuda_image_sub(g_img_b, cols, rows);

	// Copy data from host asynchronously
	cuda_image_copy_from_host_async(img_a, h_src);
	cuda_filter_copy_kernel_from_host_async(h_filter_kernel);

	// Do right processing asynchronously
	cuda_median_async(img_b, img_a, median_ksize);
	cuda_filter_async(img_a, img_b, filter_ksize);
	cuda_equalize_hist_async(img_b, img_a);

	// Copy data to host asynchronously
	cuda_image_copy_to_host_async(h_dst, img_b);
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
	cuda_filter_init();
	cuda_hist_init();

	// Allocate local buffers
	g_img_a = cuda_create_image(ColsMax, RowsMax);
	g_img_b = cuda_create_image(ColsMax, RowsMax);
}

void cuda_proc_deinit()
{
	LOG_INFO("Deinitializing CUDA proc module\n");

	// Free local buffers
	cuda_free_image(g_img_b);
	cuda_free_image(g_img_a);

	// Deinitialize modules
	cuda_hist_deinit();
	cuda_filter_deinit();

	// Deinitialize device
	cuda_reset_device();
}


Image cuda_process_host_image(
	const Image& h_src, 
	const Kernel& h_kernel, size_t median_ksize)
{
	const auto cols = h_src.cols;
	const auto rows = h_src.rows;

	LOG_INFO("Processing image with CUDA\n");
	
	// Allocate temporary host buffer
	auto h_dst = cuda_create_host_image(cols, rows);

	// Page-lock host buffers
	cuda_host_image_register(h_src);
	cuda_host_image_register(h_dst);
	cuda_host_kernel_register(h_kernel);

	// Perform processing of host image
	cuda_process_host_image_async(h_dst, h_src, h_kernel, median_ksize);
	checkCudaErrors(cudaDeviceSynchronize());

	// Un-Page-lock host buffers
	cuda_host_kernel_unregister(h_kernel);
	cuda_host_image_unregister(h_dst);
	cuda_host_image_unregister(h_src);

	return h_dst;
}