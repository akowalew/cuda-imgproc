///////////////////////////////////////////////////////////////////////////////
// cpu_filter.cpp
// 
// Implementation of CPU convolution filterer
///////////////////////////////////////////////////////////////////////////////

#include "cpu_filter.hpp"

#include <cmath>

#include "log.hpp"

void cpu_filter(Image& dst, const Image& src, const cv::Mat_<float>& kernel)
{   
    // K - the filter size
    const size_t K = kernel.rows;
    
    if (src.rows < dst.rows || src.cols < dst.cols  || dst.type() != src.type())
        throw std::string("Destination array too small or mismatching type"); // TODO add fitting exception
    
    if(kernel.rows != kernel.cols) 
        throw std::string("Mismatching filter size"); // TODO add fitting exception
    
    // Setting default offset and issuing invalid size exception
    const auto offset = K/2;
    
    if(! kernel.isContinuous() || ! src.isContinuous())
        throw std::string("Convolution kernel must be a continuous matrix");
        //ptr = kernel.clone().ptr<float>(0);  // TODO hassle handling non-continuous matrices? in GPU it has to be continuous anyways
    
    // TODO: handle situation when dst == src with intermediate copy; difficult because of possibility of destination being sparse thus cannot be easily calculated in loop
    
    float const* krnptr = kernel.ptr<float>(0);
    uchar * dstptr;
    uchar const* srcptr;
            
    srcptr = src.ptr<uchar>(0);
    
    int row_limit = src.rows - K;
    int col_limit = src.cols - K;
    int cols = src.cols;

    #pragma omp parallel for
    for(int y=0; y < row_limit; ++y) { // Y; iteration of rows
        // Destination pointer calculation - setting the right row
        dstptr = dst.ptr<uchar>(y + offset) + offset;
        
        for(int x=0; x < col_limit; ++x) { // X; iteration of columns
            // Accumulator for calculations of given round
            float acc = 0.0f;
            
            for(int dy=0; dy < K ; ++dy) // dY
                for(int dx=0; dx < K; ++dx) // dX
                    acc += *(krnptr + dy*K + dx) * (*(srcptr + (y + dy)*cols + x + dx));
            
            // Rounding and trimming the result to uchar value
            acc = round(acc);
            if(acc < 0.0f)
                acc = 0.0f;
            if(acc > 255.0f)
                acc = 255.0f;
            
            dstptr[x] = static_cast<uchar>(acc);
        }
    }
}

Image cpu_filter(const Image& src, const cv::Mat_<float>& kernel)
{
    const auto cols = src.cols;
    const auto rows = src.rows;

    LOG_INFO("Convolution filtering on CPU image %dx%d ksize %dx%d\n", 
        cols, rows, kernel.cols, kernel.rows);

    auto dst = Image(rows, cols);

    cpu_filter(dst, src, kernel);

    return dst;
}