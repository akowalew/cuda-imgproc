///////////////////////////////////////////////////////////////////////////////
// main.cpp
//
// Contains declaration of ImageProcessor class
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 22:46 CEST
///////////////////////////////////////////////////////////////////////////////

#include "ImageProcessor.hpp"

#include <cassert>

#include <memory>

#include <opencv2/highgui.hpp>

#include "imgproc/core.hpp"
#include "imgproc/filter.hpp"
#include "imgproc/histogram.hpp"

using uchar = unsigned char;

static void show_image(const char* name, const cv::Mat& mat)
{
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, mat);
}

struct ImageProcessor::Impl
{
public:
	Impl(int nrows, int ncols)
		:	m_dst(new uchar[nrows * ncols * 3])
		,	m_b(new uchar[nrows * ncols])
		,	m_b_tmp(new uchar[nrows * ncols])
		,	m_g(new uchar[nrows * ncols])
		,	m_g_tmp(new uchar[nrows * ncols])
		,	m_r(new uchar[nrows * ncols])
		,	m_r_tmp(new uchar[nrows * ncols])
		,	m_rows(nrows)
		,	m_cols(ncols)
	{
		assert(m_rows >= 0);
		assert(m_cols >= 0);
	}

	Impl(Impl&& other) = delete;

	Impl(const Impl& other) = delete;

	Impl& operator=(Impl&& other) = delete;

	Impl& operator=(const Impl& other) = delete;

	~Impl() = default;

	uchar* operator()(const uchar* src)
	{
		assert(src != nullptr);

		// Split BGR image into separate components
		imgproc::split_888(m_rows, m_cols, src, m_b.get(), m_g.get(), m_r.get());

		// Mean-Blurr 5x5 kernel
		const auto kernel_size = 5;
		const float kernel_data[kernel_size*kernel_size] = {
		    1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
		    1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
		    1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
		    1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
		    1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
		};

		// Perform mean-blurr filtering on each component
		imgproc::filter2d_8(m_rows, m_cols, m_b.get(), m_b_tmp.get(), kernel_data, kernel_size);
		imgproc::filter2d_8(m_rows, m_cols, m_g.get(), m_g_tmp.get(), kernel_data, kernel_size);
		imgproc::filter2d_8(m_rows, m_cols, m_r.get(), m_r_tmp.get(), kernel_data, kernel_size);

		// Perform histogram equalization on each component
		const auto length = (m_rows * m_cols);
		imgproc::equalize_hist_8(m_b_tmp.get(), m_b.get(), length);
		imgproc::equalize_hist_8(m_g_tmp.get(), m_g.get(), length);
		imgproc::equalize_hist_8(m_r_tmp.get(), m_r.get(), length);

		// Merge components again into BGR image
		imgproc::merge_888(m_rows, m_cols, m_b.get(), m_g.get(), m_r.get(), m_dst.get());

		return m_dst.get();
	}

private:
	std::unique_ptr<uchar> m_dst;
	std::unique_ptr<uchar> m_b;
	std::unique_ptr<uchar> m_b_tmp;
	std::unique_ptr<uchar> m_g;
	std::unique_ptr<uchar> m_g_tmp;
	std::unique_ptr<uchar> m_r;
	std::unique_ptr<uchar> m_r_tmp;
	int m_rows;
	int m_cols;
};

ImageProcessor::ImageProcessor(int nrows, int ncols)
	:	m_impl(new Impl(nrows, ncols))
{}

ImageProcessor::~ImageProcessor() = default;

uchar* ImageProcessor::operator()(const uchar* src)
{
	assert(m_impl);

	return (*m_impl)(src);
}
