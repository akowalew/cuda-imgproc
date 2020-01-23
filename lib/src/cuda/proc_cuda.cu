///////////////////////////////////////////////////////////////////////////////
// proc_cuda.cu
//
// Contains definitions of functions related to CUDA image processor module
///////////////////////////////////////////////////////////////////////////////

#include "proc_cuda.cuh"

#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

//
// Private functions
//

static void set_device(int device)
{
	printf("*** Setting CUDA device no. %d\n", device);

    checkCudaErrors(cudaSetDevice(device));
}

static void reset_device()
{
	printf("*** Resetting current CUDA device\n");

	checkCudaErrors(cudaDeviceReset());
}

//
// Public functions
//

void init_cuda()
{
	printf("*** Initializing proc CUDA module\n");

	set_device(0);
}

void deinit_cuda()
{
	printf("*** Deinitializing proc CUDA module\n");

	reset_device();
}

Image process_image_cuda(Image img, const ProcessConfig& config)
{
	printf("*** Processing image with CUDA\n");

	return img;
}
