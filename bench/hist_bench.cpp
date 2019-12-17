///////////////////////////////////////////////////////////////////////////////
// hist_bench.cpp
//
// Contains definitions of benchmarks for hist module
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.12.2019 19:46 CEST
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include "hist.hpp"

//! Performs benchmarking of equalize_hist_8 function
static void equalize_hist_8(benchmark::State& state, int width, int height)
{
    // Create source and destination image
    auto src = Image(width, height, CV_8UC1);
    auto dst = Image(src.rows, src.cols, CV_8UC1);

    // Generate random pattern on source image
    for(auto i = 0; i < src.total(); ++i)
    {
        src.data[i] = (rand() % 256);
    }

    // Perform benchmark
    for(auto _ : state)
    {
        equalize_hist_8(src, dst);
    }
}

BENCHMARK_CAPTURE(equalize_hist_8, 640x480, 640, 480)
    ->UseRealTime();
