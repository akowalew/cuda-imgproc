///////////////////////////////////////////////////////////////////////////////
// hist_cuda_v1_test.cpp
//
// Contains implementation of tests for hist_cuda_v1 module
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 22:51 CEST
///////////////////////////////////////////////////////////////////////////////

#include "doctest.h"

#include <cstdio>

#include <algorithm>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "imgproc.hpp"
#include "hist_cuda1.hpp"

TEST_SUITE("CUDA v1")
{

TEST_CASE("We can perform operations on images histograms")
{
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaDeviceReset());

	SUBCASE("Histogram may be calculated for an image")
	{
		// Parameters of source image
		const auto width = 8;
		const auto height = 8;

		// Create device-side stuff
		auto d_img = create_cuda_image(width, height);
	    auto d_hist = create_cuda_histogram();

	    // Create host-side stuff
	    auto h_hist = create_host_histogram();

	    SUBCASE("Any histogram of any image must have at least one non-zero value")
	    {
	    	// Do not set source image to have any value (we don't care here)

	        // Wait for all async operations to be done
		    checkCudaErrors(cudaDeviceSynchronize());

		    // Perform histogram calculation
	        calculate_hist_8_cuda1(d_img, d_hist);

		    checkCudaErrors(cudaDeviceSynchronize());

	        // Copy histogram from device to host
		    copy_cuda_histogram_to_host(d_hist, h_hist);

	        // Ensure, that there is at least one non-zero counter value
	        CHECK(std::any_of(h_hist.begin(), h_hist.end(),
	        	[](auto v) { return (v != 0); }));
	    }

	    SUBCASE("Image filled with one value will have histogram with only one non-zero counter")
	    {
	    	const unsigned char value = 0;

	    	// Fill source image with some value
	    	fill_cuda_image(d_img, value);

	        // Wait for all async operations to be done
		    checkCudaErrors(cudaDeviceSynchronize());

		    // Perform histogram calculation
	        calculate_hist_8_cuda1(d_img, d_hist);

	        // Wait for all async operations to be done
		    checkCudaErrors(cudaDeviceSynchronize());

	        // Copy histogram from device to host
		    copy_cuda_histogram_to_host(d_hist, h_hist);

	        // Ensure, that histogram contains only one non-zero counter
	        //  which is located on position equal to value used to fill
	        //  and which value is equal to number of pixels
	        REQUIRE(value < h_hist.size());
	        CHECK(h_hist.data[value] == (width * height));

	        CHECK(std::all_of(h_hist.begin(), std::next(h_hist.begin(), value),
	        	[](auto v) { return (v == 0); }));
	        CHECK(std::all_of(std::next(h_hist.begin(), value + 1), h_hist.end(),
	        	[](auto v) { return (v == 0); }));
	    }

	    // Free allocated host memory
	    free_host_histogram(h_hist);

        // Free allocated device memory
	    free_cuda_histogram(d_hist);
	    free_cuda_image(d_img);

	} // TEST_CASE("Histograms may be calculated for an image")

} // TEST_CASE("We can perform operations on images histograms")

} // TEST_SUITE("CUDA1")
