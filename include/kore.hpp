///////////////////////////////////////////////////////////////////////////////
// core.hpp
//
// Contains declarations of core image processing functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 17:19 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>

#include "image.hpp"

/**
 * @brief Splits 3-component image into separate components
 * @details
 *
 * @param src source, 3-component image
 * @param dst array of 1-components images
 */
void split_888(const Image& src, std::array<Image, 3>& dst);

/**
 * @brief Merges 3 separate components into one 3-component image
 * @details Each component needs to have equal size and depth
 *
 * @param src 3 components to be merged
 * @param dst destination image, 3-component
 */
void merge_888(const std::array<Image, 3>& src, Image& dst);





//Breaks some random pixels(into black pixels), to prove that median works.
void break_pixels(int part_to_be_broken, cv::Mat& r, cv::Mat& g, cv::Mat& b);


