///////////////////////////////////////////////////////////////////////////////
// hist_bench.cpp
//
// Contains definitions of benchmarks for hist module
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 13:59 CEST
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include "hist.hpp"

//! Performs benchmarking of equalize_hist function
static void equalize_hist(benchmark::State& state, int width, int height)
{
    // Create source and destination image
    auto src = Image(width, height, CV_8UC1);
    auto dst = Image(src.rows, src.cols, CV_8UC1);

    // Perform benchmark
    for(auto _ : state)
    {
        equalize_hist(src, dst);
    }
}

BENCHMARK_CAPTURE(equalize_hist, 3840x2160, 3840, 2160)
    ->UseRealTime();
