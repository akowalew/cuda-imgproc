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

//Assuming max filter size = 64x64

#define KERNELDIM 5
#define KERNELSIZE (1<<KERNELDIM) // 32

#define BLOCKSIZE 32
__constant__ float CONVKERNEL[KERNELSIZE*KERNELSIZE];


__global__ static void filter_kernel(uchar const* yin, uchar *yout, int n_rows, int n_cols, int K) {
    int c = threadIdx.x + blockIdx.x*blockDim.x;
    int r = threadIdx.y + blockIdx.y*blockDim.y;
    float acc = 0.0f;
    for(int i=0; i<K; ++i)
        for(int j=0; j<K; ++j) {
            acc += static_cast<float>(yin[(r + i)*n_cols + c + j]) * CONVKERNEL[K*i + j];
        }
    if(acc > 255.0)
        acc = 255.0f;
    if(acc < 0.0)
        acc = 0.0f;
    acc += 0.5f;
    yout[n_cols*r + c] = static_cast<uchar>(acc);
}

__host__ void filter(const Image& src, Image& dst, const Image& kernel, int offset)
{
    const size_t K = kernel.rows;
    
    // Setting default offset
    if(offset < 0)
    {
        offset = K/2;
    }

    // Ensure that offset is not higher than filter size
    assert(offset <= K);
    
    // Padding rows and cols size to kernel size granularity
    // Exact sufficient size is: ((DIM + BLOCK_SIZE-1)/BLOCK_SIZE)*BLOCK_SIZE + KERNEL_SIZE-1
    int cols = ((src.cols + KERNELSIZE - 1 + BLOCKSIZE - 1) / BLOCKSIZE) * BLOCKSIZE;
    int rows = ((src.rows + KERNELSIZE - 1 + BLOCKSIZE - 1) / BLOCKSIZE) * BLOCKSIZE;
    
    uchar *device_in;
    uchar *device_out;

    checkCudaErrors(cudaMalloc(&device_in, sizeof(uchar) * cols * rows));
    checkCudaErrors(cudaMalloc(&device_out, sizeof(uchar) * cols * rows));
        
    uchar const* srcptr;
    // Copying image data to device with padding
    for(int r=0; r < src.rows; ++r) {
        srcptr = src.ptr<uchar>(r);
        checkCudaErrors(cudaMemcpy(device_in, srcptr, sizeof(uchar)*src.cols, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(device_in + r*cols + src.cols, 0, sizeof(uchar)*(cols-src.cols)));
    }
    checkCudaErrors(cudaMemset(device_in + src.rows*cols, 0, sizeof(uchar)*((rows-src.rows)*cols)));
    
    // Copying convolution filter coefficients to __constant__ memory
    checkCudaErrors(cudaMemcpyToSymbol(kernel.ptr<float>(0), CONVKERNEL, (K*K)*sizeof(float), 0, cudaMemcpyHostToDevice));
    
    // Zero-ing the device output memory
    checkCudaErrors(cudaMemset(device_out, 0, sizeof(uchar)*rows*cols));
    
    unsigned const blocks_x = (src.rows - K + 1 + BLOCKSIZE -1) / BLOCKSIZE;
    unsigned const blocks_y = (src.cols - K + 1 + BLOCKSIZE -1) / BLOCKSIZE;

    //wywolanie kernela
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE), dimGrid(blocks_x, blocks_y);
    filter_kernel<<<dimGrid, dimBlock>>>(device_in, device_out, rows, cols, K);
    checkCudaErrors(cudaGetLastError());
    
    uchar * dstptr;
    for(int i=0; i < src.rows-K+1; ++i) {
        dstptr = dst.ptr<uchar>(i+offset) + offset;
        checkCudaErrors(cudaMemcpy(dstptr, device_out + i*cols, sizeof(uchar)*(src.cols - K + 1), cudaMemcpyDeviceToHost));
    }

    cudaFree(device_in);
    cudaFree(device_out);
}

void cuda_filter(CudaImage& dst, const CudaImage& src, CudaKernelSize ksize)
{
    // Ensure proper shapes of images
    assert(dst.cols == src.cols);
    assert(dst.rows == src.rows);
    
    auto kernel = cuda_create_mean_blurr_kernel(ksize);

    cuda_image_copy(dst, src);

    cuda_free_kernel(kernel);
}

CudaImage cuda_filter(const CudaImage& src, CudaKernelSize ksize)
{
	// Get shape of an image
	const auto cols = src.cols;
	const auto rows = src.rows;

	LOG_INFO("Convolution filtering with CUDA of image %lux%lu ksize %lu\n", cols, rows, ksize);

	// Allocate image on the device
	auto dst = cuda_create_image(cols, rows);

	// Perform convolution filtering
	cuda_filter(dst, src, ksize);

	// Return filtered image
    return dst;
}
