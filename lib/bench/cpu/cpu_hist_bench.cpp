///////////////////////////////////////////////////////////////////////////////
// cpu_hist_bench.cpp
//
// Contains definitions of benchmarks for CPU histogram equalizer
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include "cpu_proc.hpp"
#include "cpu_hist.hpp"

static void get_resolutions(benchmark::internal::Benchmark* benchmark)
{
    benchmark->Args({320, 240}); 
    benchmark->Args({640, 480});
    benchmark->Args({1024, 768});
    benchmark->Args({1920, 1080});
    benchmark->Args({2560, 1440});
    benchmark->Args({3840, 2160});
}

//! Performs benchmarking of cpu_equalize_hist function
static void cpu_equalize_hist(benchmark::State& state)
{
    const size_t cols = state.range(0);
    const size_t rows = state.range(1);

    cpu_proc_init();
    auto src = create_image(cols, rows);
    auto dst = create_image(cols, rows);

    for(const auto _ : state)
    {
        cpu_equalize_hist(dst, src);
    }

    free_image(dst);
    free_image(src);
    cpu_proc_deinit();
}

BENCHMARK(cpu_equalize_hist)
    ->UseRealTime()
    ->Apply(get_resolutions);
