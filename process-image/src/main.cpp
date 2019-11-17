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

// Windows names
constexpr auto OriginalImage = "Original image";
constexpr auto HSVImage = "HSV Image";
constexpr auto BlurredImage = "Blurred Image";

void init()
{
    printf("Initializing...\n");

    imgproc::init();

    cv::namedWindow(OriginalImage, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(HSVImage, cv::WINDOW_AUTOSIZE);
}

void deinit()
{
    printf("Deinitializing...\n");

    cv::destroyAllWindows();
    imgproc::deinit();
}

cv::Mat read_bgr_888(const char* image_path)
{
    assert(image_path != nullptr);

    printf("Reading image '%s' to the BGR...\n", image_path);

    auto bgr = cv::imread(image_path);
    cv::imshow(OriginalImage, bgr);

    assert(bgr.type() == CV_8UC3);

    return bgr;
}

cv::Mat bgr_to_hsv_888(const cv::Mat& bgr)
{
    assert(bgr.type() == CV_8UC3);

    const auto type = bgr.type();
    const auto rows = bgr.rows;
    const auto cols = bgr.cols;

    assert(rows > 0);
    assert(cols > 0);

    auto hsv = cv::Mat(rows, cols, type);
    imgproc::bgr888_to_hsv888(bgr.data, hsv.data, cols, rows);
    imshow(HSVImage, hsv);

    return hsv;
}

cv::Mat mean_blur_888(const cv::Mat& src)
{
    assert(src.type() == CV_8UC3);

    const auto type = src.type();
    const auto rows = src.rows;
    const auto cols = src.cols;

    assert(rows > 0);
    assert(cols > 0);

    auto blurred = cv::Mat(rows, cols, type);

    const auto kernel_size = 5;
    float kernel_data[kernel_size*kernel_size] = {
        1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
        1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
        1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
        1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
        1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
    };

    imgproc::filter2d_888(src.data, blurred.data, cols, rows,
        kernel_data, kernel_size);
    cv::imshow(BlurredImage, blurred);

    return blurred;
}

void process_image(const char* image_path)
{
    const auto bgr = read_bgr_888(image_path);
    const auto hsv = bgr_to_hsv_888(bgr);
    const auto blurred = mean_blur_888(hsv);

    cv::waitKey(0);
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

    init();
    process_image(image_path);
    deinit();
}
