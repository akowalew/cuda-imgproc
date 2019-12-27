#include "filter_ref.hpp"

#include <opencv2/imgproc.hpp>

void filter(const Image& src, Image& dst, const Image& kernel, int offset)
{
    // FIXME Offset not handled in reference solution
	const auto ddepth = -1; // Keep depth in destination same as in source
	cv::filter2D(src, dst, ddepth, kernel);
}