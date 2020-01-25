///////////////////////////////////////////////////////////////////////////////
// cuda_host_image.cu
//
// Contains definitions for CUDA host images manager
///////////////////////////////////////////////////////////////////////////////

#include "cuda_host_image.cuh"

HostImage cuda_create_host_image(size_t cols, size_t rows)
{
	return create_image(cols, rows);
}

void cuda_free_host_image(HostImage& h_img)
{
	free_image(h_img);	
}