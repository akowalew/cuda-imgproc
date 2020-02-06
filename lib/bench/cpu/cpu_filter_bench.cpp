///////////////////////////////////////////////////////////////////////////////
// cpu_hist_bench.cpp
//
// Contains definitions of benchmarks for CPU histogram equalizer
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include "cpu_proc.hpp"
#include "cpu_filter.hpp"

static void get_resolutions(benchmark::internal::Benchmark* benchmark)
{
    benchmark->Args({320, 240, 3}); 
    benchmark->Args({640, 480, 3});
    benchmark->Args({1024, 768, 3});
    benchmark->Args({1920, 1080, 3});
    benchmark->Args({2560, 1440, 3});
    benchmark->Args({3840, 2160, 3});


    benchmark->Args({320, 240, 7}); 
    benchmark->Args({640, 480, 7});
    benchmark->Args({1024, 768, 7});
    benchmark->Args({1920, 1080, 7});
    benchmark->Args({2560, 1440, 7});
    benchmark->Args({3840, 2160, 7});
    
    benchmark->Args({320, 240, 15}); 
    benchmark->Args({640, 480, 15});
    benchmark->Args({1024, 768, 15});
    benchmark->Args({1920, 1080, 15});
    benchmark->Args({2560, 1440, 15});
    benchmark->Args({3840, 2160, 15});
}

//! Performs benchmarking of cpu_equalize_hist function
static void cpu_filter(benchmark::State& state)
{
    const size_t cols = state.range(0);
    const size_t rows = state.range(1);
    const size_t ksize = state.range(2);

    cpu_proc_init();
    auto src = create_image(cols, rows);
    auto dst = create_image(cols, rows);

    for(const auto _ : state)
    {
        auto filter_kernel = create_mean_blurr_kernel(ksize);
        cpu_filter(dst, src, filter_kernel);
    }

    free_image(dst);
    free_image(src);
    cpu_proc_deinit();
}

BENCHMARK(cpu_filter)
    ->UseRealTime()
    ->Apply(get_resolutions);
