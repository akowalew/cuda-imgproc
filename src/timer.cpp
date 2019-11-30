/*
 * Author: Artur Dobrogowski
 * 2019-11-30
 */

#include "timer.hpp"
#include <stdio.h>
/**********************************************************/

#ifdef _WIN32

#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>

typedef LARGE_INTEGER app_timer_t;

#define timer(t_ptr) QueryPerformanceCounter(t_ptr)

void elapsed_time(app_timer_t start, app_timer_t stop,
                  unsigned long flop)
{
  double etime;
  LARGE_INTEGER clk_freq;
  QueryPerformanceFrequency(&clk_freq);
  etime = (stop.QuadPart - start.QuadPart) /
          (double) clk_freq.QuadPart;
  printf("CPU (total!) time = %.3f ms (%6.3f GFLOP/s)\n",
         etime * 1e3, 1e-9 * flop / etime);
}

#else

typedef struct timeval app_timer_t;

#define timer(t_ptr) gettimeofday(t_ptr, 0)

void elapsed_time(app_timer_t start, app_timer_t stop,
                  unsigned long flop)
{
  double etime;
  etime = 1000.0 * (stop.tv_sec  - start.tv_sec ) +
           0.001 * (stop.tv_usec - start.tv_usec);
  printf("CPU (total!) time = %.3f ms (%6.3f GFLOP/s)\n",
         etime, 1e-6 * flop / etime);
}

#endif

/**********************************************************/
