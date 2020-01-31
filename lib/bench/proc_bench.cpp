///////////////////////////////////////////////////////////////////////////////
// proc_bench.cpp
//
// Contains definitions of benchmarks for proc module
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include "proc.hpp"

static void get_resolutions(benchmark::internal::Benchmark* benchmark)
{
    benchmark->Args({320, 240, 3, 3}); 
    benchmark->Args({640, 480, 3, 3});
    benchmark->Args({1024, 768, 3, 3});
    benchmark->Args({1920, 1080, 3, 3});
    benchmark->Args({2560, 1440, 3, 3});
    benchmark->Args({3840, 2160, 3, 3});

    benchmark->Args({320, 240, 5, 5}); 
    benchmark->Args({640, 480, 5, 5});
    benchmark->Args({1024, 768, 5, 5});
    benchmark->Args({1920, 1080, 5, 5});
    benchmark->Args({2560, 1440, 5, 5});
    benchmark->Args({3840, 2160, 5, 5});

    benchmark->Args({320, 240, 7, 7}); 
    benchmark->Args({640, 480, 7, 7});
    benchmark->Args({1024, 768, 7, 7});
    benchmark->Args({1920, 1080, 7, 7});
    benchmark->Args({2560, 1440, 7, 7});
    benchmark->Args({3840, 2160, 7, 7});
}

//! Performs benchmarking of process_image function
static void process_image(benchmark::State& state)
{
    const size_t cols = state.range(0);
    const size_t rows = state.range(1);
    const size_t filter_ksize = state.range(2);
    const size_t median_ksize = state.range(3);

    proc_init();
    auto src = create_image(cols, rows);
    auto filter_kernel = create_mean_blurr_kernel(filter_ksize);

    for(const auto _ : state)
    {
        process_image(src, filter_kernel, median_ksize);
    }

    free_image(src);
    proc_deinit();
}

BENCHMARK(process_image)
    ->UseRealTime()
    ->Apply(get_resolutions);
