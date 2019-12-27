///////////////////////////////////////////////////////////////////////////////
// filter_bench.cpp
//
// Contains definitions of benchmarks for filter module
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.12.2019 19:49 CEST
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include "filter.hpp"

//! Performs benchmarking of filter2d_8 function
static void filter2d_8(benchmark::State& state, int width, int height, int ksize)
{
    // Create source and destination images and convolution kernel
    auto src = Image(width, height, CV_8UC1);
    auto dst = Image(src.rows, src.cols, CV_8UC1);
    auto kernel = Image(ksize, ksize, CV_32F);

    // Perform benchmark
    for(auto _ : state)
    {
        filter2d_8(src, dst, kernel);
    }
}

BENCHMARK_CAPTURE(filter2d_8, 640x480x9, 640, 480, 9)
    ->UseRealTime();