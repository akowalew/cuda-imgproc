///////////////////////////////////////////////////////////////////////////////
// writer.hpp
//
// Contains declaration of image writing functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 28.11.2019 23:57 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

/**
 * @brief Writes image to the file
 * @details
 *
 * @param path path of output file
 * @param image image to be written
 */
void write_image(const char* path, Image image);
