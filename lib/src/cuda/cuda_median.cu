///////////////////////////////////////////////////////////////////////////////
// cuda_median.cu
//
// Contains definitions for CUDA median filterer
///////////////////////////////////////////////////////////////////////////////

#include "cuda_median.cuh"

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "log.hpp"

//
// Private constants
//

constexpr int K = 32;

constexpr int KSizeMax = 8;

//
// Private functions
//

__global__ 
static void cuda_median_kernel(
	uchar* dst, size_t dpitch, 
	const uchar* src, size_t spitch,
	size_t cols, size_t rows, size_t ksize)
{	
	auto adr = [cols](int x, int y) -> int {
		return ((y * cols) + x);
	};

	// Nie można dynamicznie alokować w ten sposób. Alokujemy maksa.
	// max ksize = 7. 14 = 2*7;
	// (32+14)^2 to circa 2.5 kiB. Mamy lekko 16 kiB pamieci wspoldzielonej
	__shared__ uchar s_pixbuf[(K+14) * (K+14)];
	uchar hist[256] = {0};
	
	/*
	x, y - pozycja w calym obrazie
	x_, y_ - pozycja w shared mem
	xx, yy - zmienne do iterowania
	*/
	int x, y, x_, y_;

	{
  x_ = (threadIdx.y % 2) * 32 + threadIdx.x;
  x = blockIdx.x * K + x_;	//Adres w zrodle obrazu
  int yy = (threadIdx.y / 2) * 3;
  y = (blockIdx.y * K) + yy; //Zaden watek nie zaladuje wiecej niz 3 pixele.
  //watki ktore wyjezdzaja za krawedz obrazu, niczego nie laduja.
  if ((x_ < (K + (2 * ksize))) && (x < cols)) {

		if (y == rows || yy == (K + (2 * ksize))) goto sync;// wyjezdzamy za krawedz obrazu zrodlowego albo bufora.
		s_pixbuf[(yy * (K + (2 * KSizeMax))) + x_] = src[adr(x, y)];
		yy++; y++;

		if (y == rows || yy == (K + (2 * ksize))) goto sync;// wyjezdzamy za krawedz obrazu zrodlowego albo bufora.
		s_pixbuf[(yy * (K + (2 * KSizeMax))) + x_] = src[adr(x, y)];
		yy++; y++;

		if (y == rows || yy == (K + (2 * ksize))) goto sync;// wyjezdzamy za krawedz obrazu zrodlowego albo bufora.
		s_pixbuf[(yy * (K + (2 * KSizeMax))) + x_] = src[y*spitch + x];
		}
	sync: __syncthreads();//Można liczyć, mamy dane w pixbuf
	}

	//Pozycja w całym obrazie.	
	x = blockIdx.x * blockDim.x + threadIdx.x + ksize;
	y = blockIdx.y * blockDim.y + threadIdx.y + ksize;
	if((y >= rows - ksize) || (x >= cols - ksize))
	{
		return;
	}

	// Pozycja w s_pixbuf.
	x_ = threadIdx.x + ksize;
	y_ = threadIdx.y + ksize;

	// Napelniamy histogram
	for(int yy = threadIdx.y; yy <= y_ + ksize; yy++)
	{
		for(int xx = threadIdx.x; xx <= x_ + ksize; xx++)
		{
			hist[s_pixbuf[((yy * (K + 14)) + xx)]]++;
		}
	}

	// Policz ile pixeli trzeba odrzucic
	int counter = ksize;
	counter = ((counter * 2) + 1);
	counter *= counter;
	counter /= 2;

	// Szukaj pixela w histogramie
	uchar pixel = 0;
	do
	{
		counter -= hist[pixel];
		if(counter < 0)
		{
			break;
		}

		pixel++;
	}
	while(pixel != 255);

	// Pixel policzony
	dst[y*dpitch + x] = pixel;
}

void cuda_median_async(CudaImage& dst, const CudaImage& src, CudaMedianKernelSize ksize)
{
	// Ensure same sizes of images
	assert(src.cols == dst.cols);
	assert(src.rows == dst.rows);

	const auto cols = src.cols;
	const auto rows = src.rows;

	LOG_INFO("Median filtering with CUDA of image %lux%lu and ksize %lu\n", cols, rows, ksize);

	const auto x_min = ksize;
	const auto x_max = (cols - ksize);
	const auto y_min = ksize;
	const auto y_max = (rows - ksize);

	const auto dim_grid_x = K;
	const auto dim_grid_y = K;
	const auto dim_grid = dim3(dim_grid_x, dim_grid_y);

	const auto dim_block_x = ((x_max - x_min + K - 1) / K);
	const auto dim_block_y = ((y_max - y_min + K - 1) / K);
	const auto dim_block = dim3(dim_block_x, dim_block_y);

	// Launch median filtering kernel
	cuda_median_kernel<<<dim_block, dim_grid>>>(
		(uchar*)dst.data, dst.pitch,
		(const uchar*)src.data, src.pitch,
		src.cols, src.rows, ksize);

	// Check for kernel launch errors
	checkCudaErrors(cudaGetLastError());
}
