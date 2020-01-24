///////////////////////////////////////////////////////////////////////////////
// cuda_image.cuh
//
// Contains definitions for CUDA image manager
///////////////////////////////////////////////////////////////////////////////

#include "cuda_image.cuh"

#include <cassert>
#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

CudaImage cuda_create_image(size_t cols, size_t rows)
{
	printf("*** Creating CUDA image of size %lux%lu\n", cols, rows);

	// Calculate parameters needed by CUDA
	const auto width = (cols * sizeof(uchar));
	const auto height = rows;

	// Perform pitched memory allocation
	void* data;
	size_t pitch;
	checkCudaErrors(cudaMallocPitch(&data, &pitch, width, height));

	printf("*** Created CUDA image at 0x%p and pitch %lu\n", data, pitch);

	// Return created image
	return CudaImage { data, pitch, cols, rows };
}

void cuda_free_image(CudaImage& d_img)
{
	printf("*** Freeing CUDA image at %p\n", d_img.data);

	checkCudaErrors(cudaFree(d_img.data));
}

CudaImage cuda_image_clone(const CudaImage& d_src)
{
	// Retrieve device image shape
	const auto cols = d_src.cols;
	const auto rows = d_src.rows;

	printf("*** Cloning CUDA image of size %lux%lu\n", cols, rows);

	// Allocate image on the device and copy device data
	auto d_dst = cuda_create_image(cols, rows);
	cuda_image_copy(d_dst, d_src);

	// Return cloned device image
	return d_dst;
}

CudaImage cuda_image_clone_from_host(const HostImage& h_src)
{
	// Validate host image shape
	assert(h_src.cols > 0);
	assert(h_src.cols > 0);

	// Retrieve host image shape
	const auto cols = (size_t) h_src.cols;
	const auto rows = (size_t) h_src.rows;

	printf("*** Cloning CUDA image from host of size %lux%lu\n", cols, rows);

	// Allocate image on the device and copy host data
	auto d_dst = cuda_create_image(cols, rows);
	cuda_image_copy_from_host(d_dst, h_src);

	// Return cloned host device image
	return d_dst;
}

HostImage cuda_image_clone_to_host(const CudaImage& d_src)
{
	// Retrieve device image shape
	const auto cols = d_src.cols;
	const auto rows = d_src.rows;

	printf("*** Cloning CUDA image to host of size %lux%lu\n", cols, rows);

	// Allocate image on host and copy device data
	auto h_dst = create_image(cols, rows);
	cuda_image_copy_to_host(h_dst, d_src);

	// Return cloned host image
	return h_dst;
}

void cuda_image_copy(CudaImage& d_dst, const CudaImage& d_src)
{
	// Retrieve device image shape
	const auto cols = d_dst.cols;
	const auto rows = d_dst.rows;

	printf("*** Copying CUDA image of size %lux%lu\n", cols, rows);

	// Calculate parameters needed by cuda
	const auto width = (cols * sizeof(uchar));
	const auto height = rows;

	// Perform data copy on device
	checkCudaErrors(cudaMemcpy2D(d_dst.data, d_dst.pitch,
		d_src.data, d_src.pitch, width, height, cudaMemcpyDeviceToDevice));
}

void cuda_image_copy_from_host(CudaImage& d_dst, const HostImage& h_src)
{
	// Ensure proper images sizes
	assert(d_dst.cols == h_src.cols);
	assert(d_dst.rows == h_src.rows);

	cuda_image_copy_data_from_host(d_dst, h_src.data);
}

void cuda_image_copy_data_from_host(CudaImage& d_dst, const void* h_src_data)
{
	const auto cols = d_dst.cols;
	const auto rows = d_dst.rows;

	printf("*** Copying CUDA image data from host of size %lux%lu\n", cols, rows);

	// Calculate parameters needed by cuda
	const auto width = (cols * sizeof(uchar));
	const auto height = rows;

	// Source's pitch is same as row width
	const auto dpitch = d_dst.pitch;
	const auto spitch = width;

	// Perform data copy from host
	checkCudaErrors(cudaMemcpy2D(d_dst.data, dpitch, h_src_data, spitch,
		width, height, cudaMemcpyHostToDevice));
}

void cuda_image_copy_to_host(HostImage& h_dst, const CudaImage& d_src)
{
	// Ensure proper images sizes
	assert(h_dst.cols == d_src.cols);
	assert(h_dst.rows == d_src.rows);

	cuda_image_copy_data_to_host(h_dst.data, d_src);
}

void cuda_image_copy_data_to_host(void* h_dst_data, const CudaImage& d_src)
{
	const auto cols = d_src.cols;
	const auto rows = d_src.rows;

	printf("*** Copying CUDA image data to host of size %lux%lu\n", cols, rows);

	// Calculate parameters needed by cuda
	const auto width = (cols * sizeof(uchar));
	const auto height = rows;

	// Destination's pitch is same as row width
	const auto spitch = d_src.pitch;
	const auto dpitch = width;

	// Perform data copy to host
	checkCudaErrors(cudaMemcpy2D(h_dst_data, dpitch, d_src.data, spitch,
		width, height, cudaMemcpyDeviceToHost));	
}

// CudaImage::CudaImage(size_t width, size_t height)
//     :   width(width)
//     ,   height(height)
// {
//     // Allocate memory for 2D image
//     checkCudaErrors(cudaMallocPitch(&data, &pitch, width, height));
// }

// CudaImage::~CudaImage()
// {
//     if(data)
//     {
//         // Free memory of 2D image
//         checkCudaErrors(cudaFree(data));
//     }
// }

// void CudaImage::fill(int value)
// {
//     // Fill 2D image with value
// 	checkCudaErrors(cudaMemset2D(data, pitch, value, width, height));
// }

// void CudaImage::copy_from_host(const Image& img)
// {
//     // Both images must have same size
//     assert(img.cols == width);
//     assert(img.rows == height);

//     // Pitch of host image must be zero
//     assert(img.isContinuous());
//     const auto img_pitch = 0;

//     // Perform copy of image from host to device
//     checkCudaErrors(cudaMemcpy2D(data, pitch,
//         img.data, img_pitch, width, height, cudaMemcpyHostToDevice));
// }

// void CudaImage::copy_to_host(Image& img)
// {
//     // Both images must have same size
//     assert(img.cols == width);
//     assert(img.rows == height);

//     // Pitch of host image must be zero
//     assert(img.isContinuous());
//     const auto img_pitch = 0;

//     // Perform copy of image from device to host
//     checkCudaErrors(cudaMemcpy2D(img.data, img_pitch,
//         data, pitch, width, height, cudaMemcpyDeviceToHost));
// }
