///////////////////////////////////////////////////////////////////////////////
// cuda_filter.cu
//
// Contains declarations for CUDA convolution filterer
///////////////////////////////////////////////////////////////////////////////

#include "cuda_filter.cuh"

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "log.hpp"

#include "cuda_common.cuh"

//
// Private globals
//

//! Number of threads in block
static constexpr size_t K = 32;

//! Maximum size of a kernel is 16, over this size it's certain that FFT approach is faster
static constexpr size_t KSizeMax = 16;

//! Maximum number of columns in extended images
static constexpr size_t ColsExMax = (((ColsMax + (KSizeMax-1) + (K-1)) / K) * K);

//! Maximum number of rows in exyen
static constexpr size_t RowsExMax = (((RowsMax + (KSizeMax-1) + (K-1)) / K) * K);

static constexpr size_t SharedTileSize = K + KSizeMax - 1;

CudaImage g_src_ex;

//! Constant array with filter kernel coefficients
__constant__ float c_kernel[KSizeMax*KSizeMax];

//! Shared array for tile buffering
__shared__ float tile[SharedTileSize][SharedTileSize];

//
// Private functions
//

void cuda_filter_init()
{
    g_src_ex = cuda_create_image(ColsExMax, RowsExMax);
}

void cuda_filter_deinit()
{
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

__global__ static void filter_kernel(
    uchar* dst_ex, size_t dpitch,
    const uchar* src_ex, size_t spitch_ex,  
    size_t ksize)
{
//     const int x = threadIdx.x + blockIdx.x*blockDim.x;
//     const int y = threadIdx.y + blockIdx.y*blockDim.y;
    
    // Copying data to shared memory
    for(int yy = threadIdx.y; yy < K+ksize-1; yy += K)
        for(int xx = threadIdx.x; xx < K+ksize-1; xx += K)
            // TODO: sprawdzic mnozenie indeksow przez staly i zmienny rozmiar - jawnie
            // TODO: sprawdzic uzmiennienie blockDim*blockidx
            tile[yy][xx] = static_cast<float>(src_ex[(yy + blockIdx.y*blockDim.y)*spitch_ex + (xx + blockIdx.x*blockDim.x)]); 
        
    __syncthreads();
    
    // Computing the convolution for tile
    float acc = 0.0f;
    for(int i=0; i<ksize; ++i)
        for(int j=0; j<ksize; ++j)
            // TODO: test KSizeMax or ksize in c_kernel size
            acc += c_kernel[i*ksize + j] * tile[threadIdx.y + i][threadIdx.x + j];
        
    __syncthreads();
        
    if(acc > 255.0f)
    {
        acc = 255.0f;
    }
    else if(acc < 0.0f)
    {
        acc = 0.0f;
    }
    
    acc += 0.5f;
//     dst_ex[(y)*dpitch_ex + (x)] = static_cast<uchar>(acc);
    dst_ex[(threadIdx.y + blockIdx.y*blockDim.y)*dpitch + (threadIdx.x + blockIdx.x*blockDim.x)] = static_cast<uchar>(acc);
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
    const auto src_ex = cuda_image_sub(g_src_ex, cols_ex, rows_ex);

    const auto sdata = (uchar*) src.data;
    const auto ddata = (uchar*) dst.data;
    const auto spitch = src.pitch;
    const auto dpitch = dst.pitch;
    const auto sdata_ex = (uchar*) src_ex.data;
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
        ddata, dpitch, 
        sdata_ex, spitch_ex, 
        ksize);

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
}
