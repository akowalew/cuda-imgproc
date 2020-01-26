///////////////////////////////////////////////////////////////////////////////
// cuda_lut_bench.cu
//
// Contains definitions of benchmarks for CUDA histogram equalizer
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include "cuda_proc.cuh"
#include "cuda_hist.cuh"

#include "cuda_bench_common.cuh"

//! Performs benchmarking of cuda_apply_lut function
static void cuda_apply_lut(benchmark::State& state)
{
    const size_t cols = state.range(0);
    const size_t rows = state.range(1);

    cuda_proc_init();
    auto src = cuda_create_image(cols, rows);
    auto dst = cuda_create_image(cols, rows);
    auto lut = cuda_create_lut();

    cuda_benchmark(state, [&dst, &src, &lut] {
        cuda_apply_lut_async(dst, src, lut);
    });

    cuda_free_lut(lut);
    cuda_free_image(dst);
    cuda_free_image(src);
    cuda_proc_deinit();
}

BENCHMARK(cuda_apply_lut)
    ->UseRealTime()
    ->UseManualTime()
    ->Apply(get_resolutions);
