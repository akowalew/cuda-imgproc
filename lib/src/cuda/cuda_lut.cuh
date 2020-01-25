///////////////////////////////////////////////////////////////////////////////
// cuda_lut.cuh
//
// Contains declarations for CUDA LUTs manager
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_image.cuh"

constexpr auto LUTSize = 256;

struct CudaLUT
{
	using Type = uchar;

	Type* data;
};

CudaLUT cuda_create_lut();

void cuda_free_lut(CudaLUT& lut);

void cuda_apply_lut_async(CudaImage& dst, const CudaImage& src, const CudaLUT& lut);

void cuda_apply_lut(CudaImage& dst, const CudaImage& src, const CudaLUT& lut);

CudaImage cuda_apply_lut(const CudaImage& src, const CudaLUT& lut);