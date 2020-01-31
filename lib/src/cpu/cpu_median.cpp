///////////////////////////////////////////////////////////////////////////////
// cpu_median.cpp
//
// Contains implementation of CPU median filterer
///////////////////////////////////////////////////////////////////////////////

#include "cpu_median.hpp"

#include <cassert>
#include <string.h>

#include "log.hpp"

void cpu_median(Image& dst, const Image& src, int ksize)
{
	int x_min = ksize;
	int x_max = src.cols - ksize;
	int y_min = ksize;
	int y_max = src.rows - ksize;

	//ksize promien kolka w metryce miejskiej.
	//ksize 0 - jeden pixel
	//ksize 1 - 9 pixeli
	//ksize 2 - 25 pixeli
	//ksize 3 - 49 pixeli
	//ksize n - (n*2+1)^2 pixeli
	//ksize max 7 - (7*2+1) = (14+1)^2

	//rozmiar zalezy od liczby bitow na subpixel.

	auto adr = [&src](int x, int y) {
		return((int)((y * src.cols) + x));

	};

	#pragma omp parallel for
	for (int x = x_min; x < x_max; x++)
	{
		unsigned char hist[256] = {};
		for (int y = y_min; y < y_max; y++)
		{
			//wyzeruj histogram
			memset(hist, 0, 256);
			//napelnij histogram
			for (int xx = x - ksize; xx <= x + ksize; xx++) for (int yy = y - ksize; yy <= y + ksize; yy++) hist[	src.data[adr(xx, yy)] ]++;
			//znajdz odpowiedniego pixela
				//policz ile pixelai trzeba odrzucic
			unsigned char pixel = 0;
			int counter = ksize;
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
	}
}

Image cpu_median(const Image& src, int ksize)
{
	const auto cols = src.cols;
	const auto rows = src.rows;

    LOG_INFO("Median filtering on CPU image %dx%d ksize %dx%d\n", cols, rows, ksize, ksize);

    auto dst = Image(rows, cols);

    cpu_median(dst, src, ksize);

    return dst;
}