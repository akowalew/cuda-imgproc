#include <math.h>
#include <opencv2/imgproc.hpp>
#include "filter.hpp"
#include "helper_cuda.h"
#include <string>


//Assuming max filter size = 64x64

#define KERNELDIM 6
#define KERNELSIZE (1<<KERNELDIM) // 64
__constant__ float CONVKERNEL[KERNELSIZE*KERNELSIZE];

__host__ void filter2d_8(const Image& src, Image& dst, const Image& kernel, int offset)
{
    const size_t K = kernel.rows;
    if (src.rows < dst.rows || src.cols < dst.cols  || dst.type() != src.type())
        throw std::string("Destination array too small or mismatching type"); // TODO add fitting exception
    
    if(kernel.rows != kernel.cols) 
        throw std::string("Mismatching filter size"); // TODO add fitting exception
    
    // Setting default offset and issuing invalid size exception
    if(offset < 0)
        offset = K/2;
    if(offset > K)
        throw std::string("Offset higher than filter size!");
    
    if(! kernel.isContinuous() || ! src.isContinuous())
        throw std::string("Convolution kernel must be a continuous matrix");
    
//     float const* krnptr = kernel.ptr<float>(0);
//     uchar * dstptr;
    uchar const* srcptr;
//             
    srcptr = src.ptr<uchar>(0);
    
    // Padding rows and cols size to kernel size granularity
    int cols = ((src.cols + KERNELSIZE - 1) >> KERNELDIM) << KERNELDIM;
    int rows = ((src.rows + KERNELSIZE - 1) >> KERNELDIM) << KERNELDIM;
    
    float *device_in;
	float *device_out;
	float *device_convol;
    
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc(&device_in, sizeof(uchar) * cols * rows));
	checkCudaErrors(cudaMalloc(&device_out, sizeof(float) * cols * rows));
	//checkCudaErrors(cudaMalloc(&device_convol, sizeof(float)*K*K));
// 	checkCudaErrors(cudaMemcpy(device_in, yin, dim_bytes, cudaMemcpyHostToDevice));
// 	//checkCudaErrors(cudaMemcpy(device_out, yout, dim_bytes, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMemcpy(device_coeff, coeff, sizeof(float)*(n + 1), cudaMemcpyHostToDevice));
// 	
// 	unsigned const blocks = (len + K - 1) / K;
// 
// 	cudaEvent_t start, stop; // pomiar czasu wykonania jądra 
// 	checkCudaErrors(cudaEventCreate(&start));
// 	checkCudaErrors(cudaEventCreate(&stop));
// 	checkCudaErrors(cudaEventRecord(start, 0));
// 
// 	audiofir_kernel<<<blocks, K>>>(device_out, device_in, device_coeff, n, len);
// 	checkCudaErrors(cudaGetLastError());
// 	audiofir_kernel<<<blocks, K>>> (device_out+len, device_in+len, device_coeff, n, len);
// 	checkCudaErrors(cudaGetLastError());
// 
// 	checkCudaErrors(cudaEventRecord(stop, 0)); 
// 	checkCudaErrors(cudaEventSynchronize(stop));
// 	float elapsedTime; 
// 	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
// 	checkCudaErrors(cudaEventDestroy(start));
// 	checkCudaErrors(cudaEventDestroy(stop));
// 
// 	//checkCudaErrors(cudaMemcpy(yin, device_in, dim_bytes, cudaMemcpyDeviceToHost));
// 	checkCudaErrors(cudaMemcpy(yout, device_out, dim_bytes, cudaMemcpyDeviceToHost));
// 	cudaFree(device_in);
// 	cudaFree(device_out);
// 	cudaFree(device_coeff);
// 	printf("GPU (kernel) time = %.3f ms (%6.3f GFLOP/s)\n", elapsedTime, 1e-6 * 2 * ((double)n + 1) * 2 * ((double)len) / elapsedTime);
    

//     for(int y=0; y < row_limit; ++y) { // Y; iteration of rows
//         // Destination pointer calculation - setting the right row
//         dstptr = dst.ptr<uchar>(y + offset) + offset;
//         
//         for(int x=0; x < col_limit; ++x) { // X; iteration of columns
//             // Accumulator for calculations of given round
//             float acc = 0.0f;
//             
//             for(int dy=0; dy < K ; ++dy) // dY
//                 for(int dx=0; dx < K; ++dx) // dX
//                     acc += *(krnptr + dy*K + dx) * (*(srcptr + (y + dy)*cols + x + dx));
//             
//             // Rounding and trimming the result to uchar value
//             acc = round(acc);
//             if(acc < 0.0f)
//                 acc = 0.0f;
//             if(acc > 255.0f)
//                 acc = 255.0f;
//             
//             dstptr[x] = static_cast<uchar>(acc);
//         }
//     }
    
    
}

__global__ static void filter2d_8_kernel(float const* yin, float *yout, float const *coeff, int n_x, int n_y, int K, int offset) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
}

/*__global__ static void audiofir_kernel(float* yout, float* yin, float* coeff, int n, int len)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x; if (i < len) {
		float acc = 0.0f;
		for (int k = 0; k <= n; ++k) {
			if (i >= k) {
				acc = acc + yin[i - k] * coeff[k];
			}
		}
		yout[i] = acc;
	}
}
void audiofir(
	float* yout, float* yin, float* coeff, int n, int len, ...)
{
	float *device_in;
	float *device_out;
	float *device_coeff;
	unsigned const dim_bytes = sizeof(float) * len * 2;
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc(&device_in, dim_bytes));
	checkCudaErrors(cudaMalloc(&device_out, dim_bytes));
	checkCudaErrors(cudaMalloc(&device_coeff, sizeof(float)*(n+1)));
	checkCudaErrors(cudaMemcpy(device_in, yin, dim_bytes, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(device_out, yout, dim_bytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_coeff, coeff, sizeof(float)*(n + 1), cudaMemcpyHostToDevice));
	
	unsigned const blocks = (len + K - 1) / K;

	cudaEvent_t start, stop; // pomiar czasu wykonania jądra 
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	audiofir_kernel<<<blocks, K>>>(device_out, device_in, device_coeff, n, len);
	checkCudaErrors(cudaGetLastError());
	audiofir_kernel<<<blocks, K>>> (device_out+len, device_in+len, device_coeff, n, len);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaEventRecord(stop, 0)); 
	checkCudaErrors(cudaEventSynchronize(stop));
	float elapsedTime; 
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	//checkCudaErrors(cudaMemcpy(yin, device_in, dim_bytes, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(yout, device_out, dim_bytes, cudaMemcpyDeviceToHost));
	cudaFree(device_in);
	cudaFree(device_out);
	cudaFree(device_coeff);
	printf("GPU (kernel) time = %.3f ms (%6.3f GFLOP/s)\n", elapsedTime, 1e-6 * 2 * ((double)n + 1) * 2 * ((double)len) / elapsedTime);
}*/
