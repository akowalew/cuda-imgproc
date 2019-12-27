///////////////////////////////////////////////////////////////////////////////
// median.cpp
//
// Contains definitions of functions related to median image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 28.11.2019 23:23 CEST
///////////////////////////////////////////////////////////////////////////////

#include "median_cpu_multi.hpp"

#include <cassert>
#include <opencv2/imgproc.hpp>
#include <string.h>

void median(const Image& src, Image& dst, int kernel_size)
{
	int x_min = kernel_size;
	int x_max = src.cols - kernel_size;
	int y_min = kernel_size;
	int y_max = src.rows - kernel_size;

	//kernel_size promien kolka w metryce miejskiej.
	//kernel_size 0 - jeden pixel
	//kernel_size 1 - 9 pixeli
	//kernel_size 2 - 25 pixeli
	//kernel_size 3 - 49 pixeli
	//kernel_size n - (n*2+1)^2 pixeli
	//kernel_size max 7 - (7*2+1) = (14+1)^2

	//rozmiar zalezy od liczby bitow na subpixel.
	unsigned char hist[256] = {};

	auto adr = [&src](int x, int y) {
		return((int)((y * src.cols) + x));

	};

	for (int x = x_min; x < x_max; x++)	for (int y = y_min; y < y_max; y++)
	 {
		//wyzeruj histogram
		memset(hist, 0, 256);
		//napelnij histogram
		for (int xx = x - kernel_size; xx <= x + kernel_size; xx++) for (int yy = y - kernel_size; yy <= y + kernel_size; yy++) hist[	src.data[adr(xx, yy)] ]++;
		//znajdz odpowiedniego pixela
			//policz ile pixelai trzeba odrzucic
		unsigned char pixel = 0;
		int counter = kernel_size;
		counter = ((counter * 2 ) + 1);
		counter *= counter;
		counter /= 2;
		//szukaj pixela w histogramie
		do {
			counter -= hist[pixel];
			if (counter < 0) break;
			pixel++;
		} while (pixel != 255);
		//zapisz pixela;
		dst.data[adr(x, y)] = pixel;


	};

	//Dziala poprawnie za pierwsza kompilacja!!

}
