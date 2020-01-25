///////////////////////////////////////////////////////////////////////////////
// writer.cpp
// 
// Contains implementation of image writer module
///////////////////////////////////////////////////////////////////////////////

#include "writer.hpp"

#include <cstdio>
#include <cassert>

#include <opencv2/highgui.hpp>

#include "log.hpp"

void write_image(Image image, const char* path)
{
    LOG_INFO("Writing image to '%s'\n", path);

    const auto written = cv::imwrite(path, image);
    if(!written)
    {
        throw std::runtime_error("Could not write image");
    }
}