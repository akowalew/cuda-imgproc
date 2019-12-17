///////////////////////////////////////////////////////////////////////////////
// main.cpp
//
// Contains implementation of entry point to `process-image` application
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////

#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>


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


enum {
	console,
	compile
} usage_type;


int main(int argc, char** argv)
{
	//choose one
	//usage_type = console;
	usage_type = compile;



	const char *input_path, *output_path;
	
	
	
	if (usage_type == console) {
		if (argc < 3)
		{
			printf("Usage: ./process-image <input_path> <output_path>\n");
			return -1;
		}

		input_path = argv[1];
		output_path = argv[2];
	}

	else if (usage_type == compile) {
		
		input_path = "C:/Users/lab/Desktop/RIM/Projekt/sample.jpg";
		output_path = "C:/Users/lab/Desktop/RIM/Projekt/out.jpg";
	}

	else {
		printf("choose usage type");
		return -1;
	}
	 


	/*
	cv::String in1("sample.jpg");

	cv::String in2("C:/Users/lab/Desktop/RIM/Projekt/sample.jpg");

	cv::String in3("C:\\Users\\lab\\Desktop\\RIM\\Projekt\\sample.jpg");
	auto in = in3;

	printf("\n%s\n", in.c_str());
	cv::Mat image = cv::imread(in, cv::IMREAD_COLOR);
	
	
	
	printf("%d", image.size());
	
	*/
	
	
	
	
	
	
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
