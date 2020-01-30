///////////////////////////////////////////////////////////////////////////////
// cuda_lut_test.cu
//
// Contains implementation of tests for CUDA lut module
///////////////////////////////////////////////////////////////////////////////

#include "doctest/doctest.h"

#include <algorithm>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "cuda_proc.cuh"
#include "cuda_lut.cuh"

static void cuda_apply_lut_filled_test(
	CudaHostImage& h_dst, CudaImage& d_dst, 
	CudaImage& src, CudaLUT& lut,
	uchar dst_v, uchar src_v, uchar lut_v)
{
	cuda_image_fill_async(d_dst, dst_v);
	cuda_image_fill_async(src, src_v);
	cuda_lut_fill_async(lut, lut_v);
    cuda_apply_lut_async(d_dst, src, lut);
    cuda_image_copy_to_host_async(h_dst, d_dst);
    checkCudaErrors(cudaDeviceSynchronize());

    CHECK(std::all_of((const uchar*) h_dst.data, h_dst.dataend, 
    	[lut_v](uchar v) { return v == lut_v; }));
}

TEST_CASE("LUTs can be applied to images")
{
	const auto cols = 8;
	const auto rows = 8;

    cuda_proc_init();
    auto h_dst = cuda_create_host_image(cols, rows);
    auto d_dst = cuda_create_image(cols, rows);
    auto src = cuda_create_image(cols, rows);
    auto lut = cuda_create_lut();

    SUBCASE("Applying LUT with zeros should make image also zeroed")
    {
    	const auto dst_v = 0xAB;
    	const auto src_v = 0xCD;
    	const auto lut_v = 0x00;
    	cuda_apply_lut_filled_test(h_dst, d_dst, src, lut, dst_v, src_v, lut_v);
    }

    SUBCASE("Applying LUT with ones should make image also oned")
    {
    	const auto dst_v = 0xAB;
    	const auto src_v = 0xCD;
    	const auto lut_v = 0x01;
    	cuda_apply_lut_filled_test(h_dst, d_dst, src, lut, dst_v, src_v, lut_v);
    }

    SUBCASE("Applying LUT with 0xFF should make image also")
    {
    	const auto dst_v = 0xAB;
    	const auto src_v = 0xCD;
    	const auto lut_v = 0xFF;
    	cuda_apply_lut_filled_test(h_dst, d_dst, src, lut, dst_v, src_v, lut_v);
    }

    cuda_free_lut(lut);
    cuda_free_image(src);
    cuda_free_image(d_dst);
    cuda_free_host_image(h_dst);
    cuda_proc_deinit();
}
