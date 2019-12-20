///////////////////////////////////////////////////////////////////////////////
// hist_cuda1_bench.cpp
//
// Contains definitions of benchmarks for hist_cuda1 module
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 13:59 CEST
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "hist_cuda1.hpp"

//! Performs benchmarking of calculate_hist_8_cuda1
static void calculate_hist_8_cuda1(benchmark::State& state, size_t width, size_t height)
{
    // Configure CUDA device
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaDeviceReset());

    auto image = create_cuda_image(width, height);
    auto hist = create_cuda_histogram();
    
    // Wait for all async operations to be done
    checkCudaErrors(cudaDeviceSynchronize());

    // Perform benchmark
    for(auto _ : state)
    {
        calculate_hist_8_cuda1(image, hist);
        
        // Wait for all async operations to be done
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Free allocated memory
    free_cuda_image(image);
    free_cuda_histogram(hist);
}

BENCHMARK_CAPTURE(calculate_hist_8_cuda1, 3840x2160, 3840, 2160)
    ->UseRealTime();
