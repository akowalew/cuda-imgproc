///////////////////////////////////////////////////////////////////////////////
// cuda_lut.cuh
//
// Contains declarations for CUDA LUTs manager
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_image.cuh"

struct CudaLUT
{
	using Type = uchar;

	constexpr static size_t Size = 256;

	constexpr static size_t BufferSize = (Size * sizeof(Type));

	Type* data;
};

CudaLUT cuda_create_lut();

void cuda_free_lut(CudaLUT& lut);

void cuda_lut_fill_async(CudaLUT& lut, uchar value);


void cuda_lut_set_async(const CudaLUT& lut);

void cuda_apply_lut_async(CudaImage& dst, const CudaImage& src);