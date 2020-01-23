///////////////////////////////////////////////////////////////////////////////
// proc_cuda.cuh
// 
// Contains declarations for CUDA image processor module
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "proc.hpp"

/**
 * @brief Initializes image processor module
 * @details It queries for available compute devices and selects one of them
 * to use in calculations. 
 */
void init_cuda();

/**
 * @brief Deinitializes image processor module
 * @details It freeds allocated memories and releases acquired device
 */
void deinit_cuda();

/**
 * @brief Processes the image according to specified configuration
 * @details Image is processed in following steps:
 *  1) Median filtration
 *  2) Convolution filtration
 *  3) Histogram equalization
 * 
 * @param img image to process
 * @param config processing configuration
 * 
 * @return processing result
 */
Image process_image_cuda(Image img, const ProcessConfig& config);