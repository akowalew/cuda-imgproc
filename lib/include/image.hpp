///////////////////////////////////////////////////////////////////////////////
// image.hpp
//
// Contains declarations of image types classes
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2/core.hpp>

using Image = cv::Mat_<unsigned char>;

// using Cols = size_t;

// using Rows = size_t;

// using ImageValue = uchar;

// Image* make_image(Cols cols, Rows rows);

// Image* make_image(Cols cols, Rows rows, ImageValue value);

// Image* zeros(Cols cols, Rows rows);

// Image* ones(Cols cols, Rows rows);

// void free_image(Image* img);

// void get_image_info(Image* img, Cols* cols, Rows* rows);

// void get_image_data(Image* img, void* dst, size_t size);

// void set_image_data(Image* img, const void* src, size_t size);
