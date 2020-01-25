///////////////////////////////////////////////////////////////////////////////
// proc.cpp
//
// Contains definitions of functions related to image processor module
///////////////////////////////////////////////////////////////////////////////

#include "proc.hpp"

#include <cstdio>

#include "cuda_proc.cuh"

#include "log.hpp"

//
// Public functions
//

void proc_init()
{
	LOG_INFO("Initializing proc module\n");

	cuda_proc_init();
}

void proc_deinit()
{
	LOG_INFO("Deinitializing proc module\n");

	cuda_proc_deinit();
}

Image process_image(const Image& img, const ProcessConfig& config)
{
	LOG_INFO("Processing image\n");

	return cuda_process_image(img, config);
}
