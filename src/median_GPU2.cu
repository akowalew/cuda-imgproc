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
#include "helper_cuda.h" 

#include "median.hpp"



// Nie zmieniać K !!
constexpr int K = 32;
__global__ static void  median_kernel(char* src, char* dst, int kernel_size, int cols, int rows);

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



	char *src_, *dst_;
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**) &src_, sizeof(char) * size));
	checkCudaErrors(cudaMalloc((void**)&dst_, sizeof(char) * size));
	checkCudaErrors(cudaMemset(dst_, 0, size)); // To nie jest konieczne, ale przydatne w debugu.
	checkCudaErrors(cudaMemcpy(src_, src.data, (sizeof(char) * size), cudaMemcpyHostToDevice));

	median_kernel <<<block, grid >> > (src_, dst_, kernel_size, src.cols, src.rows);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(dst.data, dst_, (sizeof(char) * size), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(src_));
	checkCudaErrors(cudaFree(dst_));
	checkCudaErrors(cudaDeviceReset());

}

__global__ static void  median_kernel(char* src, char* dst, int kernel_size, int cols, int rows)
{	
	// max kernel_size = 7. 14 = 2*7;
	// (32+14)^2 to circa 2.5 kiB. Mamy lekko 16 kiB pamieci wspoldzielonej


	auto adr = [cols](int x, int y) {
		return((int)((y * cols) + x));
	};
	
	assert(kernel_size > 0);
	assert(kernel_size < 8);
	assert(K == 32);


	/*
	x, y - pozycja w calym obrazie
	x_, y_ - pozycja w shared mem
	xx, yy - zmienne do iterowania
	*/
	int x, y, x_, y_;


	//Nie można dynamicznie alokować w ten sposób. Alokuijemy maksa.
	__shared__ unsigned char pixbuf[(K+14) * (K + 14)];
	unsigned char hist[256];
	memset(hist, 0, 256);
	
	//tylko pierwsze dwa warpy. Ponieważ dotyczy całych warpow, caly warp albo skoczy, albo liczy.
	if (threadIdx.y > 1 ) goto sync;	
	
	x_ = threadIdx.y * 32 + threadIdx.x;
	x = blockIdx.x * K + x_;	//Adres w zrodle obrazu	
	
	//tymczasowe uzycie alternatywne. Adres y, ale nie pixela, tylko kernel_size ponizej niego. Liczenie adresu pixela odpowiadajacego watkowi jest ponizej i zawiera kernel_size.
	y = (blockIdx.y * K);
	//watki ktore wyjezdzaja za krawedz obrazu, niczego nie laduja.
	if ((x_ <( K + (2 * kernel_size))) && (x < cols) ){
		for (int yy = 0; yy < ( K + (2 * kernel_size)); yy++) {		
			int y__ = y + yy;
			if (y__ == rows) break;	// nie wyjezdzamy za obraz
			pixbuf[(yy * (K + 14)) + x_] =  src[adr(x, y__)];		
		}
	}

	//Można liczyć, mamy dane w pixbuf
	sync:
	__syncthreads();

	//Pozycja w całym obrazie.	
	x = blockIdx.x * K + threadIdx.x + kernel_size;
	y = blockIdx.y * K + threadIdx.y + kernel_size;
	if (y >= rows - kernel_size) return;
	if (x >= cols - kernel_size) return;
		
	//Pozycja w pixbuf.
	 x_ = threadIdx.x + kernel_size;
	 y_ = threadIdx.y + kernel_size;

	 //napelniamy histogram
	 for (int yy = threadIdx.y; yy <= y_ + kernel_size; yy++) for (int xx = threadIdx.x; xx <= x_ + kernel_size; xx++)	 hist[pixbuf[((yy * (K + 14)) + xx)]]++;

	 
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
	
	//pixel policzony




	// 46 = K + 14 = 32 + 14	
	//dst[adr(x,y)] = pixbuf[(y_ * 46) + x_];	//tylko kopiowanie
	 dst[adr(x, y)] = pixel;

}


