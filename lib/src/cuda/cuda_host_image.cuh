///////////////////////////////////////////////////////////////////////////////
// cuda_host_image.cuh
//
// Contains declarations for CUDA host images manager
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

using HostImage = Image;

HostImage cuda_create_host_image(size_t cols, size_t rows);

void cuda_free_host_image(HostImage& h_img);
