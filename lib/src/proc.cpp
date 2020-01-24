///////////////////////////////////////////////////////////////////////////////
// proc.cpp
//
// Contains definitions of functions related to image processor module
///////////////////////////////////////////////////////////////////////////////

#include "proc.hpp"

#include <cstdio>

#include "cuda_proc.cuh"

//
// Public functions
//

void proc_init()
{
	printf("*** Initializing proc module\n");

	cuda_proc_init();
}

void proc_deinit()
{
	printf("*** Deinitializing proc module\n");

	cuda_proc_deinit();
}

Image process_image(const Image& img, const ProcessConfig& config)
{
	printf("*** Processing image\n");

	return cuda_process_image(img, config);
}
