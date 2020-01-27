///////////////////////////////////////////////////////////////////////////////
// cuda_filter.cu
//
// Contains declarations for CUDA convolution filterer
///////////////////////////////////////////////////////////////////////////////

#include "cuda_filter.cuh"

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "log.hpp"

//
// Private globals
//

//! Number of threads in block
static constexpr size_t K = 32;

//! Maximum size of a kernel
static constexpr size_t KSizeMax = 32;

//! Constant array with filter kernel coefficients
__constant__ float c_kernel[KSizeMax*KSizeMax];

//! Size of current stored in constant memory kernel
static size_t ksize;

//
// Private functions
//

void cuda_set_filter_kernel(const CudaKernel& kernel)
{
    ksize = kernel.size;

    LOG_INFO("Setting CUDA kernel for filter %lux%lu\n", ksize, ksize);

    const auto buffer = (void*) kernel.data;
    const auto buffer_size = (ksize * ksize * sizeof(CudaKernel::Type));
    const auto buffer_offset = 0;
    checkCudaErrors(cudaMemcpyToSymbol(c_kernel, buffer, 
        buffer_size, buffer_offset, cudaMemcpyDeviceToDevice));
}

__global__ 
static void filter_kernel(
    uchar* dst_ex, size_t dpitch_ex,
    const uchar* src_ex, size_t spitch_ex,  
    size_t ksize)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;

    float acc = 0.0f;
    for(int i = 0; i < ksize; ++i)
    {
        for(int j = 0; j < ksize; ++j)
        {
            const auto src_v = static_cast<float>(src_ex[(i+y)*spitch_ex + (j+x)]);
            const auto kernel_v = c_kernel[i*ksize + j];
            // printf("%f ", kernel_v);
            acc += src_v * kernel_v;
        }
    }

    if(acc > 255.0f)
    {
        acc = 255.0f;
    }
    else if(acc < 0.0f)
    {
        acc = 0.0f;
    }
    
    acc += 0.5f;
    
    const auto offset = (ksize / 2);
    dst_ex[(y+offset)*dpitch_ex + (x+offset)] = static_cast<uchar>(acc);
}
 
void cuda_filter_async(CudaImage& dst, const CudaImage& src)
{
    // Ensure proper shapes of images
    assert(dst.cols == src.cols);
    assert(dst.rows == src.rows);

    const auto cols = src.cols;
    const auto rows = src.rows;

    // Setting default offset
    const auto offset = (ksize / 2);
    
    // Padding rows and cols size to kernel size granularity
    int cols_ex = ((cols + (ksize-1) + (K-1)) / K) * K;
    int rows_ex = ((rows + (ksize-1) + (K-1)) / K) * K;
    
    // Allocate extended images buffers
    auto src_ex = cuda_create_image(cols_ex, rows_ex);
    auto dst_ex = cuda_create_image(cols_ex, rows_ex);
    
    const auto sdata = (uchar*) src.data;
    const auto ddata = (uchar*) dst.data;
    const auto spitch = src.pitch;
    const auto dpitch = dst.pitch;
    const auto ddata_ex = (uchar*) dst_ex.data;
    const auto sdata_ex = (uchar*) src_ex.data;
    const auto dpitch_ex = dst_ex.pitch;
    const auto spitch_ex = src_ex.pitch;

    // Copy source buffer into extended source buffer
    checkCudaErrors(cudaMemcpy2D(
        sdata_ex + offset*spitch_ex + offset*sizeof(uchar), spitch_ex, 
        sdata, spitch, 
        cols*sizeof(uchar), rows, 
        cudaMemcpyDeviceToDevice));

    // Set unused pixels of extended source buffer to zeros (top bar)
    checkCudaErrors(cudaMemset2D(
        sdata_ex, spitch_ex, 0,
        cols_ex*sizeof(uchar), offset));

    // Set unused pixels of extended source buffer to zeros (Left bar)
    checkCudaErrors(cudaMemset2D(
        sdata_ex + offset*spitch_ex, spitch_ex, 0,
        offset*sizeof(uchar), rows));

    // Set unused pixels of extended source buffer to zeros (Right bar)
    checkCudaErrors(cudaMemset2D(
        sdata_ex + offset*spitch_ex + (offset+cols)*sizeof(uchar), spitch_ex, 0,
        (cols_ex-cols-offset)*sizeof(uchar), rows));

    // Set unused pixels of extended source buffer to zeros (bottom bar)
    checkCudaErrors(cudaMemset2D(
        sdata_ex + (offset+rows)*spitch_ex, spitch_ex, 0,
        cols_ex*sizeof(uchar), (rows_ex-rows-offset)));

    checkCudaErrors(cudaDeviceSynchronize());

    // Dimensions of each block (in threads)
    const unsigned dim_block_x = K;
    const unsigned dim_block_y = K;
    const auto dim_block = dim3(dim_block_x, dim_block_y);

    // Dimensions of a grid (in blocks)
    const unsigned dim_grid_x = ((cols+K-1) / K);
    const unsigned dim_grid_y = ((rows+K-1) / K);
    const auto dim_grid = dim3(dim_grid_x, dim_grid_y);

    LOG_INFO("dim_grid %lux%lu ex %lux%lu norm %lux%lu\n", 
        dim_grid_x, dim_grid_y,
        cols_ex, rows_ex,
        cols, rows);

    // Launch filtering kernel
    filter_kernel<<<dim_grid, dim_block>>>(
        ddata_ex, dpitch_ex, 
        sdata_ex, spitch_ex, 
        ksize);

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    
    // Copy results to destination buffer
    checkCudaErrors(cudaMemcpy2D(
        ddata, dpitch, 
        ddata_ex + offset*dpitch_ex + offset*sizeof(uchar), dpitch_ex, 
        cols*sizeof(uchar), rows, 
        cudaMemcpyDeviceToDevice));

    // Free temporary images
    cuda_free_image(dst_ex);
    cuda_free_image(src_ex);
}

void cuda_filter(CudaImage& dst, const CudaImage& src)
{
    // Launch convolution filtering asynchronously
    cuda_filter_async(dst, src);

    // Wait for device to finish
    checkCudaErrors(cudaDeviceSynchronize());
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
