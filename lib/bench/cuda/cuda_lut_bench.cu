///////////////////////////////////////////////////////////////////////////////
// cuda_lut_bench.cu
//
// Contains definitions of benchmarks for CUDA histogram equalizer
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include "cuda_proc.cuh"
#include "cuda_hist.cuh"

#include "cuda_bench_common.cuh"

static void get_resolutions(benchmark::internal::Benchmark* benchmark)
{
    benchmark->Args({320, 240}); 
    benchmark->Args({640, 480});
    benchmark->Args({1024, 768});
    benchmark->Args({1920, 1080});
    benchmark->Args({2560, 1440});
    benchmark->Args({3840, 2160});
}

//! Performs benchmarking of cuda_apply_lut function
static void cuda_apply_lut(benchmark::State& state)
{
    const size_t cols = state.range(0);
    const size_t rows = state.range(1);

    cuda_proc_init();
    auto src = cuda_create_image(cols, rows);
    auto dst = cuda_create_image(cols, rows);
    auto lut = cuda_create_lut();

    cuda_lut_set_async(lut);

    cuda_benchmark(state, [&dst, &src] {
        cuda_apply_lut_async(dst, src);
    });

    cuda_free_lut(lut);
    cuda_free_image(dst);
    cuda_free_image(src);
    cuda_proc_deinit();
}

BENCHMARK(cuda_apply_lut)
    ->UseManualTime()
    ->Apply(get_resolutions);
