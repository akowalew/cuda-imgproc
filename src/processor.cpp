///////////////////////////////////////////////////////////////////////////////
// processor.cpp
//
// Contains declaration of image processor functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 29.11.2019 1:17 CEST
///////////////////////////////////////////////////////////////////////////////

#include "processor.hpp"

#include <cassert>
#include <opencv2/highgui.hpp>

#include "kore.hpp"
#include "debug.hpp"
#include "filter.hpp"
#include "median.hpp"
#include "hist.hpp"

namespace  {

// Mean-Blurr 5x5 kernel
const auto kernel_size = cv::Size(5, 5);
const auto kernel_type = CV_32F;
float kernel_data[] {
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, };
const auto kernel = Image{kernel_size, kernel_type, kernel_data};

/**
 * @brief Helper function
 * @details Merges image components and shows final result in the window
 *
 * @param src image components
 * @param name name of the window
 */
void merge_and_show(const char* name, const std::array<Image, 3>& src)
{
	// Merge components into one image
	Image image;
	merge_888(src, image);

	// Let it be displayed
	show_image(name, image);
}

} //

Image process_image(Image image)
{
	//show_image("Original", image);

	// Allocate memory for each component
	auto components = std::array<Image, 3>{{
		Image(image.rows, image.cols, CV_8UC1),
		Image(image.rows, image.cols, CV_8UC1),
		Image(image.rows, image.cols, CV_8UC1)
	}};

	auto& b = std::get<0>(components);
	auto& g = std::get<1>(components);
	auto& r = std::get<2>(components);

	// And once again
	auto components_tmp = std::array<Image, 3>{{
		Image(image.rows, image.cols, CV_8UC1),
		Image(image.rows, image.cols, CV_8UC1),
		Image(image.rows, image.cols, CV_8UC1)
	}};

	auto& b_tmp = std::get<0>(components_tmp);
	auto& g_tmp = std::get<1>(components_tmp);
	auto& r_tmp = std::get<2>(components_tmp);

	// Split BGR image into separate components
	split_888(image, components);
	merge_and_show("Splitted", components);
	
	

	break_pixels(10, r, g, b);
	merge_and_show("Broken", components);


	// // Perform median filtering on each component
	const auto median_kernel_size = 5;
	median2d_8(b, b_tmp, median_kernel_size);
	median2d_8(g, g_tmp, median_kernel_size);
	median2d_8(r, r_tmp, median_kernel_size);
	merge_and_show("Median", components_tmp);
	/*
	// // Perform mean-blurr filtering on each component
	filter2d_8(b_tmp, b, kernel);
	filter2d_8(g_tmp, g, kernel);
	filter2d_8(r_tmp, r, kernel);
	merge_and_show("Gaussian", components);

	// // Perform histogram equalization on each component
	equalize_hist_8(b, b_tmp);
	equalize_hist_8(g, g_tmp);
	equalize_hist_8(r, r_tmp);
	merge_and_show("Histogram", components_tmp);
	


	// // Merge components again into BGR image
	merge_888(components_tmp, image);
	show_image("Result", image);
	*/
	return image;
}
