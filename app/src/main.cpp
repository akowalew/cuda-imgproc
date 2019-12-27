///////////////////////////////////////////////////////////////////////////////
// main.cpp
//
// Contains implementation of entry point to `process-image` application
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////

#include <cstdio>

#include <opencv2/highgui.hpp>

#include "debug.hpp"
#include "median.hpp"
#include "filter.hpp"
#include "hist.hpp"

/**
 * @brief Reads image from specified file in BGR format
 * @details
 *
 * @param path path to the file
 * @return read image
 */
Image read_image(const char* path)
{
#if (CV_MAJOR_VERSION < 3)
    auto image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
#else
    auto image = cv::imread(path, cv::IMREAD_GRAYSCALE);
#endif
    assert(image.type() == CV_8UC1);
    assert(image.channels() == 1);
    return image;
}

/**
 * @brief Writes image to the file
 * @details
 *
 * @param path path of output file
 * @param image image to be written
 */
void write_image(const char* path, Image image)
{
    const auto written = cv::imwrite(path, image);
    if(!written)
    {
        throw std::runtime_error("Could not write image");
    }
}

// Mean-Blurr 5x5 kernel
const auto kernel_size = cv::Size(5, 5);
const auto kernel_type = CV_32F;
float kernel_data[] {
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f,
    1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, 1.0/25.0f, };
const auto kernel = Image{kernel_size, kernel_type, kernel_data};

/**
 * @brief Executes processing pipeline on given image
 * @details Processing algorithm consists of five steps:
 * 1) Splitting BGR image into B, G and R components
 * 2) Applying median filter on each component separately
 * 3) Applying Gaussian blur on each component separately
 * 4) Equalizing histograms of each component separately
 * 5) Merging B, G and R components into final image
 *
 * @param image image to be processed
 * @return image after processing
 */
Image process_image(Image image)
{
    show_image("Original", image);

    // We need another image to store temporary results
    auto image_tmp = Image(image.rows, image.cols, CV_8UC1);

    // Perform median filtering on each component
    const auto median_kernel_size = 5;
    median(image, image_tmp, median_kernel_size);
    show_image("Median", image_tmp);

    // Perform mean-blurr filtering on each component
    filter(image_tmp, image, kernel);
    show_image("Gaussian", image);

    // Perform histogram equalization on each component
    equalize_hist(image, image_tmp);
    show_image("Histogram", image_tmp);

    image = image_tmp.clone();
    return image;
}

/**
 * @brief Main program routine
 * @details It parses command line arguments,
 * reads image from specified file, processes it and writes it
 * to specified file.
 *
 * @param argc argument counter
 * @param argv argument vector
 *
 * @return status code
 */
int main(int argc, char** argv)
{
    if(argc < 3)
    {
        printf("Usage: ./process-image <input_path> <output_path>\n");
        return 1;
    }

    const auto input_path = argv[1];
    const auto output_path = argv[2];

    try
    {
        const auto src_image = read_image(input_path);
        const auto dst_image = process_image(src_image);
        write_image(output_path, dst_image);

        wait_for_exit();
    }
    catch(std::exception& ex)
    {
        printf("Error: %s\n", ex.what());
    }
}
