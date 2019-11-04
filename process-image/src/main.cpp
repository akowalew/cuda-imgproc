#include <cstdio>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "imgproc/imgproc.hpp"
#include "imgproc/format.hpp"

// Windows names
constexpr auto OriginalImage = "Original image";
constexpr auto HSVImage = "HSV Image";

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

cv::Mat read_image_bgr(const char* image_path)
{
    assert(image_path != nullptr);

    printf("Reading image '%s' to the BGR...\n", image_path);

    auto bgr = cv::imread(image_path);
    cv::imshow(OriginalImage, bgr);

    return bgr;
}

cv::Mat convert_bgr_to_hsv(const cv::Mat& bgr)
{
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

void process_image(const char* image_path)
{
    const auto bgr = read_image_bgr(image_path);
    const auto hsv = convert_bgr_to_hsv(bgr);

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
