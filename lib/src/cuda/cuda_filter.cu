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

static constexpr size_t ColsMax = 3840;

static constexpr size_t RowsMax = 2160;

static constexpr size_t ColsExMax = (((ColsMax + (KSizeMax-1) + (K-1)) / K) * K);

static constexpr size_t RowsExMax = (((RowsMax + (KSizeMax-1) + (K-1)) / K) * K);

CudaImage g_dst_ex;
CudaImage g_src_ex;

//! Constant array with filter kernel coefficients
__constant__ float c_kernel[KSizeMax*KSizeMax];

//
// Private functions
//

void cuda_filter_init()
{
    g_src_ex = cuda_create_image(ColsExMax, RowsExMax);
    g_dst_ex = cuda_create_image(ColsExMax, RowsExMax);
}

void cuda_filter_deinit()
{
    cuda_free_image(g_dst_ex);
    cuda_free_image(g_src_ex);
}

void cuda_filter_copy_kernel_from_host_async(const Kernel& kernel)
{
    // Ensure kernel is square-sized
    assert(kernel.cols == kernel.rows);
    const auto ksize = kernel.cols;

    // Ensure kernel is smaller than maximum available
    assert(ksize <= KSizeMax);

    LOG_INFO("Setting CUDA kernel for filter %lux%lu\n", ksize, ksize);

    const auto buffer = (void*) kernel.data;
    const auto buffer_size = (ksize * ksize * sizeof(CudaKernel::Type));
    const auto buffer_offset = 0;
    checkCudaErrors(cudaMemcpyToSymbolAsync(c_kernel, buffer, 
        buffer_size, buffer_offset, cudaMemcpyHostToDevice));
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
 
void cuda_filter_async(CudaImage& dst, const CudaImage& src, size_t ksize)
{
    // Ensure proper shapes of images
    assert(dst.cols == src.cols);
    assert(dst.rows == src.rows);

    const auto cols = src.cols;
    const auto rows = src.rows;
    
    // Padding rows and cols size to kernel size granularity
    const auto cols_ex = ((cols + (ksize-1) + (K-1)) / K) * K;
    const auto rows_ex = ((rows + (ksize-1) + (K-1)) / K) * K;
    
    // Obtain extended images buffers
    const auto dst_ex = cuda_image_sub(g_dst_ex, cols_ex, rows_ex);
    const auto src_ex = cuda_image_sub(g_src_ex, cols_ex, rows_ex);

    const auto sdata = (uchar*) src.data;
    const auto ddata = (uchar*) dst.data;
    const auto spitch = src.pitch;
    const auto dpitch = dst.pitch;
    const auto ddata_ex = (uchar*) dst_ex.data;
    const auto sdata_ex = (uchar*) src_ex.data;
    const auto dpitch_ex = dst_ex.pitch;
    const auto spitch_ex = src_ex.pitch;

    // Offset from begin of extended image buffer to "right" image
    const auto offset = (ksize / 2);

    // Copy source buffer into extended source buffer asynchronously
    checkCudaErrors(cudaMemcpy2DAsync(
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

    // Dimensions of each block (in threads)
    const auto dim_block_x = K;
    const auto dim_block_y = K;
    const auto dim_block = dim3(dim_block_x, dim_block_y);

    // Dimensions of a grid (in blocks)
    const auto dim_grid_x = ((cols+K-1) / K);
    const auto dim_grid_y = ((rows+K-1) / K);
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
    checkCudaErrors(cudaMemcpy2DAsync(
        ddata, dpitch, 
        ddata_ex + offset*dpitch_ex + offset*sizeof(uchar), dpitch_ex, 
        cols*sizeof(uchar), rows, 
        cudaMemcpyDeviceToDevice));
}
