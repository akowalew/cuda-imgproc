///////////////////////////////////////////////////////////////////////////////
// cuda_hist_test.cu
//
// Contains implementation of tests for CUDA hist module
///////////////////////////////////////////////////////////////////////////////

#include "doctest.h"

#include <algorithm>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "cuda_proc.cuh"
#include "cuda_hist.cuh"

static void cuda_calculate_hist_filled_test(
    CudaHistogram::Type* h_hist_data, CudaHistogram& d_hist, CudaImage& img, 
    CudaImage::Type img_v, size_t nelems)
{
    GIVEN("Image filled with 0x00's")
    {
        cuda_image_fill_async(img, img_v);

        WHEN("Calculating a histogram")
        {
            cuda_calculate_hist_async(d_hist, img);
            checkCudaErrors(cudaDeviceSynchronize());

            THEN("All counters have 0's, except one, which equals to nelems")
            {
                cuda_histogram_copy_data_to_host(h_hist_data, d_hist);
                CHECK(std::all_of(h_hist_data, h_hist_data + img_v,
                    [](uint v) { return (v == 0); }));
                CHECK(h_hist_data[img_v] == nelems);
                CHECK(std::all_of(h_hist_data + img_v + 1, h_hist_data + CudaHistogram::Size,
                    [](uint v) { return (v == 0); }));
            }
        }
    }
}

SCENARIO("Histograms may be calculated for images")
{
	const auto cols = 8;
	const auto rows = 8;
    const auto nelems = (cols * rows);

    cuda_proc_init();
    auto h_hist_data = (CudaHistogram::Type*) malloc(CudaHistogram::BufferSize);
    auto d_hist = cuda_create_histogram();
    auto img = cuda_create_image(cols, rows);

    cuda_calculate_hist_filled_test(h_hist_data, d_hist, img, 0x00, nelems);
    cuda_calculate_hist_filled_test(h_hist_data, d_hist, img, 0x01, nelems);
    cuda_calculate_hist_filled_test(h_hist_data, d_hist, img, 0xFF, nelems);

    free(h_hist_data);
    cuda_free_histogram(d_hist);
    cuda_free_image(img);
    cuda_proc_deinit();
}
