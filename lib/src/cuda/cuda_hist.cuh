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


void cuda_hist_init();

void cuda_hist_deinit();


CudaHistogram cuda_create_histogram();

void cuda_free_histogram(CudaHistogram& hist);


void cuda_calculate_hist_async(CudaHistogram& hist, const CudaImage& img);

void cuda_calculate_hist(CudaHistogram& hist, const CudaImage& img);

CudaHistogram cuda_calculate_hist(const CudaImage& src);


void cuda_gen_equalize_lut_async(CudaLUT& lut, const CudaHistogram& hist);

void cuda_gen_equalize_lut(CudaLUT& lut, const CudaHistogram& hist);

CudaLUT cuda_gen_equalize_lut(const CudaHistogram& hist);


void cuda_equalize_hist_async(CudaImage& dst, const CudaImage& src);

void cuda_equalize_hist(CudaImage& dst, const CudaImage& src);

/**
 * @brief Performs histogram equalization
 * @details 
 * 
 * @param src source image
 * @return image with equalized image
 */
CudaImage cuda_equalize_hist(const CudaImage& src);
