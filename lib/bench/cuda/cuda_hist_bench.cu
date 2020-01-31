///////////////////////////////////////////////////////////////////////////////
// cuda_hist_bench.cu
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

//! Performs benchmarking of cuda_equalize_hist function
static void cuda_equalize_hist(benchmark::State& state)
{
    const size_t cols = state.range(0);
    const size_t rows = state.range(1);

    cuda_proc_init();
    auto src = cuda_create_image(cols, rows);
    auto dst = cuda_create_image(cols, rows);

    cuda_benchmark(state, [&src, &dst] {
        cuda_equalize_hist_async(dst, src);
    });

    cuda_free_image(dst);
    cuda_free_image(src);
    cuda_proc_deinit();
}

BENCHMARK(cuda_equalize_hist)
    ->UseManualTime()
    ->Apply(get_resolutions);

//! Performs benchmarking of cuda_calculate_hist function
static void cuda_calculate_hist(benchmark::State& state)
{
    const size_t cols = state.range(0);
    const size_t rows = state.range(1);

    cuda_proc_init();
    auto img = cuda_create_image(cols, rows);
    auto hist = cuda_create_histogram();

    cuda_benchmark(state, [&img, &hist] {
        cuda_calculate_hist_async(hist, img);
    });

    cuda_free_histogram(hist);
    cuda_free_image(img);
    cuda_proc_deinit();
}

BENCHMARK(cuda_calculate_hist)
    ->UseManualTime()
    ->Apply(get_resolutions);

//! Performs benchmarking of cuda_gen_equalize_lut function
static void cuda_gen_equalize_lut(benchmark::State& state)
{
    cuda_proc_init();
    auto lut = cuda_create_lut();
    auto hist = cuda_create_histogram();

    cuda_benchmark(state, [&lut, &hist] {
        cuda_gen_equalize_lut_async(lut, hist);
    });

    cuda_free_histogram(hist);
    cuda_free_lut(lut);
    cuda_proc_deinit();
}

BENCHMARK(cuda_gen_equalize_lut)
    ->UseManualTime();
