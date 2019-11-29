///////////////////////////////////////////////////////////////////////////////
// main.cpp
//
// Contains implementation of entry point to `process-image` application
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////

#include <cstdio>

#include "debug.hpp"
#include "processor.hpp"
#include "reader.hpp"
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
