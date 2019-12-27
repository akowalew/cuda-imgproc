///////////////////////////////////////////////////////////////////////////////
// median_bench.cpp
//
// Contains definitions of benchmarks for median module
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.12.2019 19:46 CEST
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include "median.hpp"

//! Performs benchmarking of median_8 function
static void median_8(benchmark::State& state, int width, int height, int ksize)
{
    // Create source and destination image
    auto src = Image(width, height, CV_8UC1);
    auto dst = Image(src.rows, src.cols, CV_8UC1);

    // Perform benchmark
    for(auto _ : state)
    {
        median2d_8(src, dst, ksize);
    }
}

BENCHMARK_CAPTURE(median_8, 640x480x9, 640, 480, 9)
    ->UseRealTime();
