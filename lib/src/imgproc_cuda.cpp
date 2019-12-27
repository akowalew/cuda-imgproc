///////////////////////////////////////////////////////////////////////////////
// imgproc_cuda.cpp
//
// Contains definitions of functions related to imgproc library
// CUDA implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 27.12.2019 12:10 CEST
///////////////////////////////////////////////////////////////////////////////

#include "imgproc_cuda.hpp"

#include <cuda_runtime.h>

#include <helper_cuda.h>

void init()
{
    // Configure CUDA device, at the moment, by default using dev 0
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaDeviceReset()); 
}

void deinit()
{
	// Nothing to do
}