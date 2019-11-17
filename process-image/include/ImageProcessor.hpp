///////////////////////////////////////////////////////////////////////////////
// main.cpp
//
// Contains declaration of ImageProcessor class
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

class ImageProcessor
{
public:
	ImageProcessor(int nrows, int ncols);

	~ImageProcessor();

	unsigned char* operator()(const unsigned char* src);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};
