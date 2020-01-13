///////////////////////////////////////////////////////////////////////////////
// hist_cuda_v1_bench.cpp
//
// Contains definitions of benchmarks for hist_cuda_v1 module
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 13:59 CEST
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "imgproc.hpp"
#include "hist_cuda.cuh"

//! Performs benchmarking of calculate_hist
static void calculate_hist(benchmark::State& state, size_t width, size_t height)
{
    // init();

    // CudaCDF::Counter* cdf_min;

    // // Create CUDA device variables
    // auto image = CudaImage(width, height);
    // auto histogram = CudaHistogram();
    // auto lut = CudaLUT();
    // checkCudaErrors(cudaMalloc(&cdf_min, sizeof(CudaCDF::Counter)));

    // checkCudaErrors(cudaDeviceSynchronize());

    // // Perform benchmark
    // for(auto _ : state)
    // {
    //     // Do histogram equalization
    //     equalize_hist(image, histogram, histogram, cdf_min, lut, image);

    //     // Wait for all async operations to be done
    //     checkCudaErrors(cudaDeviceSynchronize());
    // }

    // // Release CUDA device variables
    // checkCudaErrors(cudaFree(cdf_min));

    // deinit();
}

BENCHMARK_CAPTURE(calculate_hist, 320x240, 320, 240)
    ->UseRealTime();
