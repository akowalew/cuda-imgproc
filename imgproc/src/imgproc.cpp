///////////////////////////////////////////////////////////////////////////////
// imgproc.cpp
//
// Contains global declarations for imgproc library
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 4.11.2019 18:31 CEST
///////////////////////////////////////////////////////////////////////////////

#include "imgproc/imgproc.hpp"

#include <cstdio>

namespace imgproc {

void init()
{
    printf("[imgproc] Initialized\n");
}

void deinit()
{
    printf("[imgproc] Deinitialized\n");
}

} // namespace imgproc
