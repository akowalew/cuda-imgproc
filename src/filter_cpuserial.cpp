#include <math.h>
#include <opencv2/imgproc.hpp>
#include "filter.hpp"


/**
 * @brief General convolution filter of arbitrary dimension
 * 
 * @param src source of image data
 * @param dst destination to write image data
 * @param kernel convolution matrix to multiply by, float parameter
 * @param offset decides offset of convolution result in kernel coordinates.
 * Values < 0 defaults to middle of the filter (kernel.rows/2).
 */
void filter2d_8(Image const& src, Image& dst, Image const& kernel, int offset)
{   
    // K - the filter size
    const size_t K = kernel.rows;
    
	if (src.rows < dst.rows || src.cols < dst.cols  || dst.type() != src.type())
        throw std::string("Destination array too small or mismatching type"); // TODO add fitting exception
    
    if(kernel.rows != kernel.cols) 
        throw std::string("Mismatching filter size"); // TODO add fitting exception
    
    // Setting default offset and issuing invalid size exception
    if(offset < 0)
        offset = K/2;
    if(offset > K)
        throw std::string("Offset higher than filter size!");
    
    if(! kernel.isContinuous() || ! src.isContinuous())
        throw std::string("Convolution kernel must be a continuous matrix");
        //ptr = kernel.clone().ptr<float>(0);  // TODO hassle handling non-continuous matrices? in GPU it has to be continuous anyways
    
    // TODO: handle situation when dst == src with intermediate copy; difficult because of possibility of destination being sparse thus cannot be easily calculated in loop
//     ImageU8 dst_ref(dst);
//     if(src.ptr<uchar const>() == dst.ptr<uchar>())
//         dst_ref = &Image(dst);
//     else
//         dst_ref = &dst;
    
    float const* krnptr = kernel.ptr<float>(0);
    uchar * dstptr;
    uchar const* srcptr;
            
    srcptr = src.ptr<uchar>(0);
    
    int row_limit = src.rows - K;
    int col_limit = src.cols - K;
    int cols = src.cols;
    
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

