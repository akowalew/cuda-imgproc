///////////////////////////////////////////////////////////////////////////////
// cuda_host_image.cuh
//
// Contains declarations for CUDA host images manager
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

using CudaHostImage = Image;

CudaHostImage cuda_create_host_image(size_t cols, size_t rows);

void cuda_free_host_image(CudaHostImage& h_img);

void cuda_host_image_register(const CudaHostImage& h_img);

void cuda_host_image_unregister(const CudaHostImage& h_img);