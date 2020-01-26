///////////////////////////////////////////////////////////////////////////////
// cuda_filter.cu
//
// Contains declarations for CUDA convolution filterer
///////////////////////////////////////////////////////////////////////////////

#include "cuda_filter.cuh"

#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "log.hpp"

constexpr size_t KERNELSIZE = 32;

constexpr size_t K = 32;

static __constant__ float CONVKERNEL[KERNELSIZE*KERNELSIZE];

static size_t ksize;

void cuda_set_filter_kernel(const CudaKernel& kernel)
{
    ksize = kernel.size;
    const auto buffer = (void*) kernel.data;
    const auto buffer_size = (ksize * ksize * sizeof(CudaKernel::Type));
    const auto buffer_offset = 0;
    checkCudaErrors(cudaMemcpyToSymbol(CONVKERNEL, buffer, 
        buffer_size, buffer_offset, cudaMemcpyHostToDevice));
}

__global__ 
static void filter_kernel(
    uchar const* yin, uchar *yout, 
    int n_rows, int n_cols, int ksize)
{
    int c = threadIdx.x + blockIdx.x*blockDim.x;
    int r = threadIdx.y + blockIdx.y*blockDim.y;
    float acc = 0.0f;

    for(int i = 0; i < ksize; ++i)
    {
        for(int j = 0; j < ksize; ++j)
        {
            acc += static_cast<float>(yin[(r + i)*n_cols + c + j]) * CONVKERNEL[ksize*i + j];
        }
    }

    if(acc > 255.0)
    {
        acc = 255.0f;
    }
    if(acc < 0.0)
    {
        acc = 0.0f;
    }
    
    acc += 0.5f;
    
    yout[n_cols*r + c] = static_cast<uchar>(acc);
}

__host__ void filter(const Image& src, Image& dst, const Image& kernel, int offset)
{
    // Setting default offset
    if(offset < 0)
    {
        offset = ksize/2;
    }

    // Ensure that offset is not higher than filter size
    assert(offset <= ksize);
    
    // Padding rows and cols size to kernel size granularity
    // Exact sufficient size is: ((DIM + BLOCK_SIZE-1)/BLOCK_SIZE)*BLOCK_SIZE + KERNEL_SIZE-1
    int cols = ((src.cols + KERNELSIZE - 1 + K - 1) / K) * K;
    int rows = ((src.rows + KERNELSIZE - 1 + K - 1) / K) * K;
    
    uchar *device_in;
    uchar *device_out;

    checkCudaErrors(cudaMalloc(&device_in, sizeof(uchar) * cols * rows));
    checkCudaErrors(cudaMalloc(&device_out, sizeof(uchar) * cols * rows));
        
    uchar const* srcptr;
    // Copying image data to device with padding
    for(int r = 0; r < src.rows; ++r)
    {
        srcptr = src.ptr<uchar>(r);
        checkCudaErrors(cudaMemcpy(device_in, srcptr, sizeof(uchar)*src.cols, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(device_in + r*cols + src.cols, 0, sizeof(uchar)*(cols-src.cols)));
    }

    checkCudaErrors(cudaMemset(device_in + src.rows*cols, 0, sizeof(uchar)*((rows-src.rows)*cols)));
    
    // Zero-ing the device output memory
    checkCudaErrors(cudaMemset(device_out, 0, sizeof(uchar)*rows*cols));
    
    unsigned const blocks_x = (src.rows - ksize + 1 + K - 1) / K;
    unsigned const blocks_y = (src.cols - ksize + 1 + K - 1) / K;

    //wywolanie kernela
    dim3 dimBlock(K, K);
    dim3 dimGrid(blocks_x, blocks_y);
    filter_kernel<<<dimGrid, dimBlock>>>(device_in, device_out, rows, cols, ksize);
    checkCudaErrors(cudaGetLastError());
    
    uchar* dstptr;
    for(int i = 0; i < src.rows - ksize + 1; ++i) 
    {
        dstptr = dst.ptr<uchar>(i+offset) + offset;
        checkCudaErrors(cudaMemcpy(dstptr, device_out + i*cols, sizeof(uchar)*(src.cols - ksize + 1), cudaMemcpyDeviceToHost));
    }

    cudaFree(device_in);
    cudaFree(device_out);
}

void cuda_filter(CudaImage& dst, const CudaImage& src)
{
    // Ensure proper shapes of images
    assert(dst.cols == src.cols);
    assert(dst.rows == src.rows);
    
    // TODO: Launch filtering kernel
    cuda_image_copy(dst, src);
}

void cuda_filter(CudaImage& dst, const CudaImage& src, const CudaKernel& kernel)
{
    // Copy filter kernel to constant memory
    cuda_set_filter_kernel(kernel);

    // Launch convolution filtering
    cuda_filter(dst, src);
}

CudaImage cuda_filter(const CudaImage& src, const CudaKernel& kernel)
{
	// Get shape of an image
	const auto cols = src.cols;
	const auto rows = src.rows;

	LOG_INFO("Convolution filtering with CUDA of image %lux%lu ksize %lu\n", cols, rows, kernel.size);

	// Allocate image on the device
	auto dst = cuda_create_image(cols, rows);

	// Perform convolution filtering
	cuda_filter(dst, src, kernel);

	// Return filtered image
    return dst;
}
