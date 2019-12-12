///////////////////////////////////////////////////////////////////////////////
// filter.cpp
//
// Contains definitions of functions related to image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////



//Tu jest gruby przypal(albo cienki, tylko C++ generuje zjebne bledy.)
// 
/*

przy includowaniu w takiej kolejnosci:
	filter
	timer
	imgproc

Paskudne bledy. Pozycja stdio nie ma znaczenia.
Przy includowaniu tej trojki w dowolnej innej kolejnosci, wszystko jest okej, z stdio na poczatku albo koncu.
Proponuje przyjac konwencje, ze najpierw includujemy <> a potem ""
Zmienilem tez nazwe core na kore, zeby nie bylo konfliktu nazw pomiedzy naszym projektem, a opencv.
*/

#include <stdio.h>
#include <opencv2/imgproc.hpp>
#include "filter.hpp"
#include "timer.hpp"


void filter2d_8(const Image& src, Image& dst, const Image& kernel) {
#ifdef DEBUG
    printf("Convolution filter call: ");
    app_timer_t start, stop;
    timer(&start);
#endif
    
    _filter2d_8(src, dst, kernel);
    
#ifdef DEBUG
    timer(&stop);
    elapsed_time(start, stop, 2 * src.cols * src.rows * kernel.cols * kernel.rows);
#endif
}
