///////////////////////////////////////////////////////////////////////////////
// cuda_lut.cuh
//
// Contains declarations for CUDA LUTs manager
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_image.cuh"

constexpr auto LUTSize = 256;

struct CUDALUT
{
	using Type = uchar;

	Type* data;
};

CUDALUT cuda_create_lut();

void cuda_free_lut(CUDALUT& lut);

void cuda_apply_lut(CudaImage& dst, const CudaImage& src, const CUDALUT& lut);

CudaImage cuda_apply_lut(const CudaImage& src, const CUDALUT& lut);