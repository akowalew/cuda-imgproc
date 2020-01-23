///////////////////////////////////////////////////////////////////////////////
// proc.cu
//
// Contains definitions of functions related to proc module
///////////////////////////////////////////////////////////////////////////////

#include "proc.hpp"

#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

void init()
{
    // Configure CUDA device, at the moment, by default using dev 0
    checkCudaErrors(cudaSetDevice(0));
}

void deinit()
{
    // Release all resources acquired on the device
    checkCudaErrors(cudaDeviceReset());
}

Image process_image(Image img, const ProcessConfig& config)
{
	printf("*** Processing image\n");

	return img;
}
