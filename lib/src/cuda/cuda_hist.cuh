///////////////////////////////////////////////////////////////////////////////
// cuda_hist.cuh
//
// Contains declarations for CUDA histograms equalizer
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_image.cuh"
#include "cuda_lut.cuh"

//
// Public declarations
//

constexpr size_t HistogramSize = 256;

struct CudaHistogram
{
	using Type = uint;

	Type* data;
};

CudaHistogram cuda_create_histogram();

void cuda_free_histogram(CudaHistogram& hist);


void cuda_calculate_hist(CudaHistogram& hist, const CudaImage& img);

CudaHistogram cuda_calculate_hist(const CudaImage& src);


void cuda_gen_equalize_lut(CUDALUT& lut, const CudaHistogram& hist);

CUDALUT cuda_gen_equalize_lut(const CudaHistogram& hist);


void cuda_equalize_hist(CudaImage& dst, const CudaImage& src);

/**
 * @brief Performs histogram equalization
 * @details 
 * 
 * @param src source image
 * @return image with equalized image
 */
CudaImage cuda_equalize_hist(const CudaImage& src);
