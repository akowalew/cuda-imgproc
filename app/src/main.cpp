///////////////////////////////////////////////////////////////////////////////
// main.cpp
//
// Contains implementation of entry point to `process-image` application
///////////////////////////////////////////////////////////////////////////////

#include <cstdio>

#include <string>

#include "debug.hpp"
#include "image.hpp"
#include "reader.hpp"
#include "proc.hpp"
#include "writer.hpp"
#include "kernel.hpp"

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
    if(argc < 5)
    {
        printf("Usage: ./process-image <src_path> <dst_path> <median_ksize> <filter_ksize>\n");
        return 1;
    }

    try
    {
        // Parse CLI arguments
        const auto src_path = argv[1];
        const auto dst_path = argv[2];
        const auto median_ksize = std::stoul(argv[3]);
        const auto filter_ksize = std::stoul(argv[4]);

        // Initialize processing library
        proc_init();

        // Do app logic
        auto filter_kernel = create_mean_blurr_kernel(filter_ksize);
        auto src = read_image(src_path);
        auto dst = process_image(src, filter_kernel, median_ksize);
        write_image(dst, dst_path);

        // Show results and wait for keyboard press
        show_image(src, "Source image");
        show_image(dst, "Destination image");
        wait_for_exit();

        // Free memory
        free_image(dst);
        free_image(src);
        free_kernel(filter_kernel);

        // Deinitialize processing library
        proc_deinit();
    }
    catch(std::exception& ex)
    {
        printf("Error: %s\n", ex.what());
    }
}
