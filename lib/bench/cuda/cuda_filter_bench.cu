#include <benchmark/benchmark.h>

#include "cuda_proc.cuh"
#include "cuda_filter.cuh"

#include "cuda_bench_common.cuh"

/**
 * @brief Provides arguments for benchmark when testing convolution filter
 * @details To use them in the benchmark then, you have to type:
 * ```c++
 *  static void some_benchmark(benchmark::State& state)
 *  {
 *      const auto cols = state.range(0);
 *      const auto rows = state.range(1);
 *      const auto ksize = state.range(2);
 *      ... 
 *  }      
 * ```
 * 
 * @param benchmark benchmark instance
 */

static void filter_test_arguments(benchmark::internal::Benchmark* benchmark)
{
    benchmark->Args({320, 240, 3}); 
    benchmark->Args({640, 480, 3});
    benchmark->Args({1024, 768, 3});
    benchmark->Args({1920, 1080, 3});
    benchmark->Args({2560, 1440, 3});
    benchmark->Args({3840, 2160, 3});

    benchmark->Args({320, 240, 5}); 
    benchmark->Args({640, 480, 5});
    benchmark->Args({1024, 768, 5});
    benchmark->Args({1920, 1080, 5});
    benchmark->Args({2560, 1440, 5});
    benchmark->Args({3840, 2160, 5});

    benchmark->Args({320, 240, 7}); 
    benchmark->Args({640, 480, 7});
    benchmark->Args({1024, 768, 7});
    benchmark->Args({1920, 1080, 7});
    benchmark->Args({2560, 1440, 7});
    benchmark->Args({3840, 2160, 7});

}

//! Performs benchmarking of cuda_median function
static void cuda_filter(benchmark::State& state)
{
    const size_t cols = state.range(0);
    const size_t rows = state.range(1);
    const size_t ksize = state.range(2);

    cuda_proc_init();
    auto src = cuda_create_image(cols, rows);
    auto dst = cuda_create_image(cols, rows);

    // Could copy the kernel to multiply by
    cuda_benchmark(state, [&dst, &src, ksize] {
        cuda_filter_async(dst, src, ksize);
    });

    cuda_free_image(dst);
    cuda_free_image(src);
    cuda_proc_deinit();
}

BENCHMARK(cuda_filter)
    ->UseManualTime()
    ->Apply(filter_test_arguments);
