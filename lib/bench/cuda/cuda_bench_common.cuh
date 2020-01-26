///////////////////////////////////////////////////////////////////////////////
// cuda_bench_common.cuh
//
// Contains common stuff for benchmarking CUDA proc module routines
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cuda_runtime.h>

#include <helper_cuda.h>

template<typename TFunc>
static void cuda_benchmark(benchmark::State& state, TFunc&& func)
{
    cudaEvent_t start;
    cudaEvent_t stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    float time_ms;

    for(auto _ : state)
    {
        checkCudaErrors(cudaEventRecord(start));

        func();

        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));
        checkCudaErrors(cudaDeviceSynchronize());

        state.SetIterationTime(time_ms / 1000.0f);
    }  

    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
}

/**
 * @brief Provides to benchmark common resolutions when testing image algorithms
 * @details To use them in the benchmark then, you have to type:
 * ```c++
 *  static void some_benchmark(benchmark::State& state)
 *  {
 *      const auto cols = state.range(0);
 *      const auto rows = state.range(1);
 *      ... 
 *  }      
 * ```
 * 
 * @param benchmark benchmark instance
 */
static void get_resolutions(benchmark::internal::Benchmark* benchmark)
{
    benchmark->Args({320, 240}); 
    benchmark->Args({640, 480});
    benchmark->Args({1024, 768});
    benchmark->Args({1920, 1080});
    benchmark->Args({2560, 1440});
    benchmark->Args({3840, 2160});
}
