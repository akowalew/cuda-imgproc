///////////////////////////////////////////////////////////////////////////////
// cuda_filter.cu
//
// Contains declarations for CUDA convolution filterer
///////////////////////////////////////////////////////////////////////////////

#include "cuda_filter.cuh"

#include <cstdio>

CudaImage cuda_filter(const CudaImage& src, CudaFilterKernelSize ksize)
{
	printf("*** Convolution filtering with CUDA\n");

    return cuda_image_clone(src);
}

// //Assuming max filter size = 64x64

// #define KERNELDIM 5
// #define KERNELSIZE (1<<KERNELDIM) // 32

// #define BLOCKSIZE 32
// __constant__ float CONVKERNEL[KERNELSIZE*KERNELSIZE];


// __global__ static void filter_kernel(uchar const* yin, uchar *yout, int n_rows, int n_cols, int K) {
//     int c = threadIdx.x + blockIdx.x*blockDim.x;
//     int r = threadIdx.y + blockIdx.y*blockDim.y;
//     float acc = 0.0f;
//     for(int i=0; i<K; ++i)
//         for(int j=0; j<K; ++j) {
//             acc += static_cast<float>(yin[(r + i)*n_cols + c + j]) * CONVKERNEL[K*i + j];
//         }
//     if(acc > 255.0)
//         acc = 255.0f;
//     if(acc < 0.0)
//         acc = 0.0f;
//     acc += 0.5f;
//     yout[n_cols*r + c] = static_cast<uchar>(acc);
// }

// __host__ void filter(const Image& src, Image& dst, const Image& kernel, int offset)
// {
//     const size_t K = kernel.rows;
//     if (src.rows < dst.rows || src.cols < dst.cols  || dst.type() != src.type())
//         throw std::string("Destination array too small or mismatching type"); // TODO add fitting exception
    
//     if(kernel.rows != kernel.cols) 
//         throw std::string("Mismatching filter size"); // TODO add fitting exception
    
//     // Setting default offset and issuing invalid size exception
//     if(offset < 0)
//         offset = K/2;
//     if(offset > K)
//         throw std::string("Offset higher than filter size!");
    
//     if(! kernel.isContinuous() || ! src.isContinuous())
//         throw std::string("Convolution kernel must be a continuous matrix");
    
// //     float const* krnptr = kernel.ptr<float>(0);
// //             
    
//     // Padding rows and cols size to kernel size granularity
//     // Exact sufficient size is: ((DIM + BLOCK_SIZE-1)/BLOCK_SIZE)*BLOCK_SIZE + KERNEL_SIZE-1
//     int cols = ((src.cols + KERNELSIZE - 1 + BLOCKSIZE - 1) / BLOCKSIZE) * BLOCKSIZE;
//     int rows = ((src.rows + KERNELSIZE - 1 + BLOCKSIZE - 1) / BLOCKSIZE) * BLOCKSIZE;
    
//     uchar *device_in;
//     uchar *device_out;
// //     float *device_convol;

//     checkCudaErrors(cudaSetDevice(0));
//     checkCudaErrors(cudaMalloc(&device_in, sizeof(uchar) * cols * rows));
//     checkCudaErrors(cudaMalloc(&device_out, sizeof(uchar) * cols * rows));
        
//     uchar const* srcptr;
//     // Copying image data to device with padding
//     for(int r=0; r < src.rows; ++r) {
//         srcptr = src.ptr<uchar>(r);
//         checkCudaErrors(cudaMemcpy(device_in, srcptr, sizeof(uchar)*src.cols, cudaMemcpyHostToDevice));
//         checkCudaErrors(cudaMemset(device_in + r*cols + src.cols, 0, sizeof(uchar)*(cols-src.cols)));
//     }
//     checkCudaErrors(cudaMemset(device_in + src.rows*cols, 0, sizeof(uchar)*((rows-src.rows)*cols)));
    
//     // Copying convolution filter coefficients to __constant__ memory
//     checkCudaErrors(cudaMemcpyToSymbol(kernel.ptr<float>(0), CONVKERNEL, (K*K)*sizeof(float), 0, cudaMemcpyHostToDevice));
    
//     // Zero-ing the device output memory
//     checkCudaErrors(cudaMemset(device_out, 0, sizeof(uchar)*rows*cols));
    
//     // Initialising cuda Events for calculation time measurement
//     cudaEvent_t start, stop;
//     checkCudaErrors(cudaEventCreate(&start));
//     checkCudaErrors(cudaEventCreate(&stop));
//     checkCudaErrors(cudaEventRecord(start, 0));
    
    
//     unsigned const blocks_x = (src.rows - K + 1 + BLOCKSIZE -1) / BLOCKSIZE;
//     unsigned const blocks_y = (src.cols - K + 1 + BLOCKSIZE -1) / BLOCKSIZE;
//     //wywolanie kernela
//     dim3 dimBlock(BLOCKSIZE, BLOCKSIZE), dimGrid(blocks_x, blocks_y);
//     filter_kernel<<<dimGrid, dimBlock>>>(device_in, device_out, rows, cols, K);
//     checkCudaErrors(cudaGetLastError());

//     // Measuring the elapsed time
//     checkCudaErrors(cudaEventRecord(stop, 0)); 
//     checkCudaErrors(cudaEventSynchronize(stop));
//     float elapsedTime; 
//     checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
//     checkCudaErrors(cudaEventDestroy(start));
//     checkCudaErrors(cudaEventDestroy(stop));

    
//     uchar * dstptr;
//     for(int i=0; i < src.rows-K+1; ++i) {
//         dstptr = dst.ptr<uchar>(i+offset) + offset;
//         checkCudaErrors(cudaMemcpy(dstptr, device_out + i*cols, sizeof(uchar)*(src.cols - K + 1), cudaMemcpyDeviceToHost));
//     }
//     cudaFree(device_in);
//     cudaFree(device_out);
    
// //  printf("GPU (kernel) time = %.3f ms (%6.3f GFLOP/s)\n", elapsedTime, 1e-6 * 2 * ((double)n + 1) * 2 * ((double)len) / elapsedTime);
// }
