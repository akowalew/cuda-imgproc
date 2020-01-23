///////////////////////////////////////////////////////////////////////////////
// hist.hpp
//
// Contains declarations of functions working on images histograms
///////////////////////////////////////////////////////////////////////////////

#pragma once

//
// Forward declarations
//

struct Image;

//
// Public declarations
//

struct Histogram;

/**
 * @brief Calculates histogram of given image
 * @details 
 * 
 * @param hist histogram to calculate
 * @param img source image
 */
void calculate_hist(Histogram* hist, const Image* img);

Histogram* calculate_hist(const Image* img);



void equalize_hist(Image* dst, const Image* src);

Image* equalize_hist(const Image* src);