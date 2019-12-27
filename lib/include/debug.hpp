///////////////////////////////////////////////////////////////////////////////
// debug.hpp
//
// Contains declaration of image debug functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 29.11.2019 1:17 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

/**
 * @brief Sends image to be displayed on window with given name
 * @details Provided image is cloned inside in order
 * to may use same image instance for multiple windows (e.g. after
 * executing multiple algorithms in place on one image)
 *
 * @param name window name
 * @param image image to be displayed
 */
void show_image(const char* name, Image image);

/**
 * @brief Hangs until any key is pressed
 * @details
 */
void wait_for_exit();
