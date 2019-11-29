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

#include <experimental/array>

#include <opencv2/highgui.hpp>

#include "core.hpp"
#include "debug.hpp"
#include "filter.hpp"
#include "median.hpp"
#include "hist.hpp"

// Mean-Blurr 5x5 kernel
const auto kernel_size = cv::Size(5, 5);
const auto kernel_type = CV_32F;
float kernel_data[] {
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
};

const auto kernel = Image{kernel_size, kernel_type, kernel_data};

Image process_image(Image image)
{
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

	assert(b.type() == image.depth());
	assert(g.type() == image.depth());
	assert(r.type() == image.depth());
	assert(image.type() == CV_8UC3);

	// Split BGR image into separate components
	split_888(image, components);

	// // Perform median filtering on each component
	const auto median_kernel_size = 5;
	median2d_8(b, b_tmp, median_kernel_size);
	median2d_8(g, g_tmp, median_kernel_size);
	median2d_8(r, r_tmp, median_kernel_size);

	// // Perform mean-blurr filtering on each component
	filter2d_8(b_tmp, b, kernel);
	filter2d_8(g_tmp, g, kernel);
	filter2d_8(r_tmp, r, kernel);

	// // Perform histogram equalization on each component
	equalize_hist_8(b, b_tmp);
	equalize_hist_8(g, g_tmp);
	equalize_hist_8(r, r_tmp);

	// // Merge components again into BGR image
	merge_888(components_tmp, image);

	return image;
}

// struct ImageProcessor::Impl
// {
// public:
// 	Impl(int nrows, int ncols)
// 		:	m_dst(new uchar[nrows * ncols * 3])
// 		,	m_b(new uchar[nrows * ncols])
// 		,	m_b_tmp(new uchar[nrows * ncols])
// 		,	m_g(new uchar[nrows * ncols])
// 		,	m_g_tmp(new uchar[nrows * ncols])
// 		,	m_r(new uchar[nrows * ncols])
// 		,	m_r_tmp(new uchar[nrows * ncols])
// 		,	m_rows(nrows)
// 		,	m_cols(ncols)
// 	{
// 		assert(m_rows >= 0);
// 		assert(m_cols >= 0);
// 	}

// 	uchar* operator()(const uchar* src)
// 	{
// 		assert(src != nullptr);


// 		return m_dst.get();
// 	}

// private:
// 	std::unique_ptr<uchar> m_dst;
// 	std::unique_ptr<uchar> m_b;
// 	std::unique_ptr<uchar> m_b_tmp;
// 	std::unique_ptr<uchar> m_g;
// 	std::unique_ptr<uchar> m_g_tmp;
// 	std::unique_ptr<uchar> m_r;
// 	std::unique_ptr<uchar> m_r_tmp;
// 	int m_rows;
// 	int m_cols;
// };

