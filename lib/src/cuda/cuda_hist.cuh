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

struct CudaHistogram
{
	using Type = uint;

	constexpr static size_t Size = 256;

	constexpr static size_t BufferSize = (Size * sizeof(Type));

	Type* data;
};


void cuda_hist_init();

void cuda_hist_deinit();


CudaHistogram cuda_create_histogram();

void cuda_free_histogram(CudaHistogram& hist);

void cuda_histogram_fill_async(CudaHistogram& hist, CudaHistogram::Type value);

void cuda_histogram_copy_data_to_host(CudaHistogram::Type* h_data, CudaHistogram& d_hist);


void cuda_calculate_hist_async(CudaHistogram& hist, const CudaImage& img);

void cuda_gen_equalize_lut_async(CudaLUT& lut, const CudaHistogram& hist);


/**
 * @brief Performs histogram equalization
 * @details 
 * 
 * @param src source image
 * @return image with equalized image
 */
void cuda_equalize_hist_async(CudaImage& dst, const CudaImage& src);
