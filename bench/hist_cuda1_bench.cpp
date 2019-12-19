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

#include "hist.hpp"

//! Performs benchmarking of equalize_hist_8 CUDA1 version
static void equalize_hist_8_cuda1(benchmark::State& state, int width, int height)
{
    // Create source and destination image
    auto src = Image(width, height, CV_8UC1);
    auto dst = Image(src.rows, src.cols, CV_8UC1);

    // Generate random pattern on source image
    for(auto i = 0; i < src.total(); ++i)
    {
        src.data[i] = (rand() % 256);
    }

    // Configure CUDA device
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaDeviceSynchronize());



    // Perform benchmark
    for(auto _ : state)
    {
        // equalize_hist_8(src, dst);
    }
}

BENCHMARK_CAPTURE(equalize_hist_8_cuda1, 3840x2160, 3840, 2160)
    ->UseRealTime();
