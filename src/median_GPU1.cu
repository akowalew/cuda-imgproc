///////////////////////////////////////////////////////////////////////////////
// median.cpp
//
// Contains definitions of functions related to median image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 28.11.2019 23:23 CEST
///////////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <string.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_API.h"

#include "device_launch_parameters.h"
//#include <stdio.h>
#include "helper_cuda.h" 

#include "median.hpp"

//Kompletnie sie nie oplaca robic mniejszych.
constexpr int K = 32;
__global__ static void  median_kernel(unsigned char* src, unsigned char* dst, int kernel_size, int cols, int rows);

//Mocno polegam na tym, ze jeden pixel(subpixel) to unsigned char.
void median2d_8(const Image& src, Image& dst, int kernel_size) {
	//kernel_size max 7 - (7*2+1) = (14+1)^2
	//7+32+7 = 32 + 14
	int x_min = kernel_size;
	int x_max = src.cols - kernel_size;
	int y_min = kernel_size;
	int y_max = src.rows - kernel_size;
	int size = src.cols * src.rows;

	//Rozmiar jednego gridu.

	dim3 grid(K, K, 1);
	dim3 block(  (x_max - x_min + K -1) / K, (y_max - y_min + K - 1) / K, 1);



	unsigned char *src_, *dst_;
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**) &src_, sizeof(unsigned char) * size));
	checkCudaErrors(cudaMalloc((void**)&dst_, sizeof(unsigned char) * size));
	checkCudaErrors(cudaMemset(dst_, 0, size)); // To nie jest konieczne, ale przydatne w debugu.
	checkCudaErrors(cudaMemcpy(src_, src.data, (sizeof(unsigned char) * size), cudaMemcpyHostToDevice));

	median_kernel <<<block, grid >> > (src_, dst_, kernel_size, src.cols, src.rows);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(dst.data, dst_, (sizeof(unsigned char) * size), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(src_));
	checkCudaErrors(cudaFree(dst_));
	checkCudaErrors(cudaDeviceReset());

}

//Stara funkcja do skanibalizowania
void median2d(const Image& src, Image& dst, int kernel_size)
{
	int x_min = kernel_size;
	int x_max = src.cols - kernel_size;
	int y_min = kernel_size;
	int y_max = src.rows - kernel_size;

	//kernel_size promien kolka w metryce miejskiej.
	//kernel_size 0 - jeden pixel
	//kernel_size 1 - 9 pixeli
	//kernel_size 2 - 25 pixeli
	//kernel_size 3 - 49 pixeli
	//kernel_size n - (n*2+1)^2 pixeli
	//kernel_size max 7 - (7*2+1) = (14+1)^2

	//rozmiar zalezy od liczby bitow na subpixel.
	unsigned char hist[256] = {};

	auto adr = [&src](int x, int y) {
		return((int)((y * src.cols) + x));

	};

	for (int x = x_min; x < x_max; x++)	for (int y = y_min; y < y_max; y++)
	{
		//wyzeruj histogram
		memset(hist, 0, 256);
		//napelnij histogram
		for (int xx = x - kernel_size; xx <= x + kernel_size; xx++) for (int yy = y - kernel_size; yy <= y + kernel_size; yy++) hist[src.data[adr(xx, yy)]]++;
		//znajdz odpowiedniego pixela
			//policz ile pixelai trzeba odrzucic
		unsigned char pixel = 0;
		int counter = kernel_size;
		counter = ((counter * 2) + 1);
		counter *= counter;
		counter /= 2;
		//szukaj pixela w histogramie
		do {
			counter -= hist[pixel];
			if (counter < 0) break;
			pixel++;
		} while (pixel != 255);
		//zapisz pixela;
		dst.data[adr(x, y)] = pixel;


	};

	

}


//KERNEL kopiujacy, bez ramki

__global__ static void  median_kernell(unsigned char* src, unsigned char* dst, int kernel_size, int cols, int rows)
{	int x, y;

	x = blockIdx.x * K + threadIdx.x + kernel_size;
	y = blockIdx.y * K + threadIdx.y + kernel_size;
	int point = y * cols + x;
	if ((x < cols - kernel_size) && (y < rows - kernel_size)) dst[point] = src[point];
}




//KERNEL

__global__ static void  median_kernel(unsigned char * src, unsigned char* dst, int kernel_size, int cols, int rows)
{	


	auto adr = [cols](int x, int y) {
		return((int)((y * cols) + x));
	};

	

	//wyzeruj histogram
	unsigned char hist[256];
	memset(hist, 0, 256);

	int x = blockIdx.x * K + threadIdx.x + kernel_size;
	int y = blockIdx.y * K + threadIdx.y + kernel_size;
	if (y > rows - kernel_size) return;
	if (x > cols - kernel_size) return;
	
	
		
	//napelnij histogram
	for (int yy = y - kernel_size; yy <= y + kernel_size; yy++) for (int xx = x - kernel_size; xx <= x + kernel_size; xx++) 
		hist[src[adr(xx, yy)]]++;
		
	//znajdz odpowiedniego pixela
			//policz ile pixelai trzeba odrzucic
		
		int counter = kernel_size;
		counter = ((counter * 2) + 1);
		counter *= counter;
		counter /= 2;
		//szukaj pixela w histogramie
		unsigned char pixel = 0;
		do {
			counter -= hist[pixel];
			if (counter < 0) break;
			pixel++;
		} while (pixel != 255);
		//zapisz pixela;
		dst[adr(x, y)] = pixel;
}


















































