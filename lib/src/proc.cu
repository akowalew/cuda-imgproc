///////////////////////////////////////////////////////////////////////////////
// proc.cu
//
// Contains definitions of functions related to proc module
///////////////////////////////////////////////////////////////////////////////

#include "proc.hpp"

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

void init()
{
	printf("*** Initializing proc module\n");

	set_device(0);
}

void deinit()
{
	printf("*** Deinitializing proc module\n");

	reset_device();
}

Image process_image(Image img, const ProcessConfig& config)
{
	printf("*** Processing image\n");

	return img;
}
