///////////////////////////////////////////////////////////////////////////////
// main.cpp
//
// Contains implementation of entry point to `process-image` application
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////

#include <cstdio>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "imgproc/imgproc.hpp"
#include "imgproc/format.hpp"
#include "imgproc/filter.hpp"
#include "imgproc/histogram.hpp"

#include "ImageProcessor.hpp"

void show_image(const char* name, const cv::Mat& mat)
{
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, mat);
}

cv::Mat read_image(const char* image_path)
{
    assert(image_path != nullptr);

    printf("Reading image '%s'...\n", image_path);
    auto src = cv::imread(image_path);
    assert(src.type() == CV_8UC3);
    show_image("src", src);

    return src;
}

cv::Mat process_image(const cv::Mat& src)
{
    const auto data = src.data;
    const auto rows = src.rows;
    const auto cols = src.cols;

    ImageProcessor image_processor(rows, cols);
    auto dst_data = image_processor(data);
    auto dst = cv::Mat(rows, cols, CV_8UC3, dst_data);
    show_image("dst", dst);

    return dst;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("Usage: ./process-image <image_path>\n");
        return 1;
    }

    assert(argv[1] != nullptr);
    const auto image_path = argv[1];
    const auto src = read_image(image_path);
    const auto dst = process_image(src);

    cv::waitKey(0);
}
