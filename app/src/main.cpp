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
        const auto process_config = ProcessConfig {
            MedianKernelSize{std::stoul(argv[3])},
            FilterKernelSize{std::stoul(argv[4])},
            FilterKernelType::MeanBlurr
        };

        // Initialize app
        init();

        // Do app logic
        const auto src = read_image(src_path);
        const auto dst = process_image(src, process_config);
        write_image(dst, dst_path);

        wait_for_exit();

        // Deinitialize app
        deinit();
    }
    catch(std::exception& ex)
    {
        printf("Error: %s\n", ex.what());
    }
}
