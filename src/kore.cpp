///////////////////////////////////////////////////////////////////////////////
// core.cpp
//
// Contains definitions of core image processing functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////



#include <cassert>
#include <opencv2/imgproc.hpp>
#include <stdlib.h>
#include "kore.hpp"

void split_888(const Image& src, std::array<Image, 3>& dst)
{
	auto dst_array = std::vector<Image>{dst[0], dst[1], dst[2]};
    cv::split(src, dst_array);
}

void merge_888(const std::array<Image, 3>& src, Image& dst)
{
	auto src_array = std::vector<Image>{src[0], src[1], src[2]};
	cv::merge(src_array, dst);
}


#include <cassert>


using cv::Mat;
void break_pixels(int part_to_be_broken , Mat &r, Mat &g, Mat &b)
{
	//part_to_be_broken parts per thousand
	int ptb = part_to_be_broken % 1000;
	int rows = r.rows;
	int cols = r.cols;
	//wszystkie 3 obrazy musza miec identyczny rozmiar.
	assert(rows == g.rows);
	assert(cols == g.cols);
	assert(rows == b.rows);
	assert(cols == b.cols);
	int size = rows * cols;
	ptb = ((size * ptb) / 1000);
	
	for (int i = 0; i < ptb; i++) {
		//RAND_MAX 2^16-1, 15 bits.
		int point = rand();
		int x2    = rand();		
		point = (point << 15) ^ x2;
		point = point % size;
		r.data[point] = 0;
		g.data[point] = 0;
		b.data[point] = 0;
	}






}
