///////////////////////////////////////////////////////////////////////////////
// reader.hpp
//
// Contains declaration of image reading functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 28.11.2019 23:55 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

/**
 * @brief Reads image from specified file in BGR format
 * @details
 *
 * @param path path to the file
 * @return read image
 */
Image read_image(const char* path);
