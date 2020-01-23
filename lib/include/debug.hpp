///////////////////////////////////////////////////////////////////////////////
// debug.hpp
//
// Contains declaration of image debug functions
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

/**
 * @brief Sends image to be displayed on window with given name
 * @details Provided image is cloned inside in order
 * to may use same image instance for multiple windows (e.g. after
 * executing multiple algorithms in place on one image)
 *
 * @param image image to be displayed
 * @param name window name
 */
void show_image(Image image, const char* name);

/**
 * @brief Hangs until any key is pressed
 * @details
 */
void wait_for_exit();
