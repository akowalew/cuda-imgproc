///////////////////////////////////////////////////////////////////////////////
// proc.cpp
//
// Contains definitions of functions related to image processor module
///////////////////////////////////////////////////////////////////////////////

#include "proc.hpp"

#include <cstdio>

#include "proc_cuda.cuh"

//
// Public functions
//

void init()
{
	printf("*** Initializing proc module\n");

	init_cuda();
}

void deinit()
{
	printf("*** Deinitializing proc module\n");

	deinit_cuda();
}

Image process_image(Image img, const ProcessConfig& config)
{
	printf("*** Processing image\n");

	return process_image_cuda(img, config);
}
