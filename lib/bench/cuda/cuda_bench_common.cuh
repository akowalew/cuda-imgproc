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
