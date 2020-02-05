
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

//! Performs benchmarking of cuda_median function
static void cuda_filter(benchmark::State& state)
{
    const size_t cols = state.range(0);
    const size_t rows = state.range(1);
    const size_t ksize = state.range(2);

    //Initialising simple filter
//     cv::Mat_<float> filter(16,16);
//     float v = 1/static_cast<float>(ksize*ksize);
//     for(int i=0; i<ksize*ksize; ++i)
//         *(filter.ptr<float>(0)) = v;
        
    //Copying real kernel for processing matters mostly for the conditional clamping part of computing. Should use real test data with it as well though
//     cuda_filter_copy_kernel_from_host_async(filter);
    cuda_benchmark(state, [&dst, &src, ksize] {
        cuda_filter_async(dst, src, ksize);
    });

}

BENCHMARK(cuda_filter)
    ->UseManualTime()
    ->Apply(filter_test_arguments);
