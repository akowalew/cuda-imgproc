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

void split_888(const Image& src, std::array<Image, 3>& dst);

void merge_888(const std::array<Image, 3>& src, Image& dst);
