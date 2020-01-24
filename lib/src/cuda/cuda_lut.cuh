///////////////////////////////////////////////////////////////////////////////
// lut.hpp
//
// Contains declarations of functions working on images histograms
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "types.hpp"

//
// Forward declarations
//

struct Image;

//
// Public declarations
//

struct LUT;

using LUTValue = uchar;

LUT* make_lut();

LUT* make_lut(LUTValue value);

void free_lut(LUT* lut);

void get_lut_data(const LUT* lut, void* data, size_t size);

void set_lut_data(LUT* lut, const void* data, size_t size);

void bind_lut(const LUT* lut);

void apply_lut(Image* dst, const Image* src);

Image* apply_lut(const Image* src, const LUT* lut);
