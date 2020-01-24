///////////////////////////////////////////////////////////////////////////////
// image.cpp
//
// Contains declarations for image manager
///////////////////////////////////////////////////////////////////////////////

#include "image.hpp"

Image create_image(size_t cols, size_t rows)
{
	printf("*** Creating image of size %lux%lu\n", cols, rows);

	return cv::Mat(rows, cols, CV_8UC1);
}

void free_image(Image& img)
{
	img.release();
}