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
#include "cuda_median.cuh"

//
// Private functions
//

static void cuda_set_device(int device)
{
	printf("*** Setting CUDA device no. %d\n", device);

    checkCudaErrors(cudaSetDevice(device));
}

static void cuda_reset_device()
{
	printf("*** Resetting current CUDA device\n");

	checkCudaErrors(cudaDeviceReset());
}

static HostImage cuda_process_host_image(const HostImage& h_src, const ProcessConfig& config)
{
	printf("*** Processing host image with CUDA\n");

	auto d_src = cuda_image_clone_from_host(h_src);
	auto d_medianed = cuda_median(d_src, config.median_ksize);
	auto d_filtered = cuda_filter(d_medianed, config.filter_ksize);
	auto d_equalized = cuda_equalize_hist(d_filtered);
	auto h_dst = cuda_image_clone_to_host(d_equalized);

	cuda_free_image(d_equalized);
	cuda_free_image(d_filtered);
	cuda_free_image(d_medianed);
	cuda_free_image(d_src);

	return h_dst;
}

//
// Public functions
//

void cuda_init()
{
	printf("*** Initializing CUDA proc module\n");

	cuda_set_device(0);
}

void cuda_deinit()
{
	printf("*** Deinitializing CUDA proc module\n");

	cuda_reset_device();
}

Image cuda_process_image(const Image& img, const ProcessConfig& config)
{
	return cuda_process_host_image(img, config);
}
