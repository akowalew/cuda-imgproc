///////////////////////////////////////////////////////////////////////////////
// cuda_image.cuh
//
// Contains declarations for CUDA image manager
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"
#include "cuda_host_image.cuh"

struct CudaImage
{
	void* data;
	size_t pitch;
	size_t cols;
	size_t rows;
};

CudaImage cuda_create_image(size_t cols, size_t rows);

void cuda_free_image(CudaImage& d_img);


CudaImage cuda_image_clone(const CudaImage& d_src);

CudaImage cuda_image_clone_from_host(const CudaHostImage& h_src);

CudaHostImage cuda_image_clone_to_host(const CudaImage& d_src);


void cuda_image_copy(CudaImage& d_dst, const CudaImage& d_src);

void cuda_image_copy_from_host(CudaImage& d_dst, const CudaHostImage& h_src);

void cuda_image_copy_data_from_host(CudaImage& d_dst, const void* h_src_data);

void cuda_image_copy_to_host(CudaHostImage& h_dst, const CudaImage& d_src);

void cuda_image_copy_data_to_host(void* h_dst_data, const CudaImage& d_src);


void cuda_image_fill_async(CudaImage& img, uchar value);