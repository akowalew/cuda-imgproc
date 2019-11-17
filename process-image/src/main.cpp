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

void show_image(const char* name, const cv::Mat& mat)
{
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, mat);
}

void init()
{
    printf("Initializing...\n");

    imgproc::init();
}

void deinit()
{
    printf("Deinitializing...\n");

    cv::destroyAllWindows();
    imgproc::deinit();
}

cv::Mat read_bgr(const char* image_path)
{
    assert(image_path != nullptr);

    printf("Reading image '%s' to the BGR...\n", image_path);

    auto bgr = cv::imread(image_path);

    assert(bgr.type() == CV_8UC3);

    return bgr;
}

cv::Mat bgr_to_hsv(const cv::Mat& bgr)
{
    assert(bgr.type() == CV_8UC3);

    const auto type = bgr.type();
    const auto rows = bgr.rows;
    const auto cols = bgr.cols;

    assert(rows > 0);
    assert(cols > 0);

    auto hsv = cv::Mat(rows, cols, type);
    imgproc::bgr888_to_hsv888(bgr.data, hsv.data, cols, rows);

    return hsv;
}

cv::Mat mean_blur(const cv::Mat& src)
{
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

    imgproc::filter2d_8(src.data, blurred.data, cols, rows,
        kernel_data, kernel_size);

    return blurred;
}

cv::Mat equalize_hist(const cv::Mat& src)
{
    assert(src.type() == CV_8UC1);

    const auto type = src.type();
    const auto rows = src.rows;
    const auto cols = src.cols;

    assert(rows > 0);
    assert(cols > 0);

    auto equalized = cv::Mat(rows, cols, type);

    imgproc::equalize_hist_8(src.data, equalized.data, src.total());

    return equalized;
}

std::vector<cv::Mat> split_image(const cv::Mat& src)
{
    auto splitted = std::vector<cv::Mat>(src.channels());

    cv::split(src, splitted);

    return splitted;
}

cv::Mat merge_images(const std::vector<cv::Mat>& images)
{
    auto merged = cv::Mat();

    cv::merge(images, merged);

    return merged;
}

void process_image(const char* image_path)
{
    const auto bgr = read_bgr(image_path);
    show_image("BGR", bgr);

    const auto splitted = split_image(bgr);
    const auto& b = splitted[0];
    const auto& g = splitted[1];
    const auto& r = splitted[2];
    show_image("b", b);
    show_image("g", g);
    show_image("r", r);

    const auto b_blurred = mean_blur(b);
    show_image("b_blurred", b_blurred);

    const auto b_equalized = equalize_hist(b_blurred);
    show_image("b_equalized", b_equalized);

    const auto g_blurred = mean_blur(g);
    show_image("g_blurred", g_blurred);

    const auto g_equalized = equalize_hist(g_blurred);
    show_image("g_equalized", g_equalized);

    const auto r_blurred = mean_blur(r);
    show_image("r_blurred", r_blurred);

    const auto r_equalized = equalize_hist(r_blurred);
    show_image("r_equalized", r_equalized);

    const auto images = std::vector<cv::Mat>{b_equalized, g_equalized, r_equalized};
    const auto merged = merge_images(images);
    show_image("merged", merged);

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
