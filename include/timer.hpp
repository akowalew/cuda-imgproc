/*
 * Author: Artur Dobrogowski
 * 2019-11-30
 */

#ifndef TIMER_H
#define TIMER_H

/**********************************************************/

#ifdef _WIN32

#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>

typedef LARGE_INTEGER app_timer_t;

#define timer(t_ptr) QueryPerformanceCounter(t_ptr)
#else
#include <sys/time.h>
typedef struct timeval app_timer_t;

#define timer(t_ptr) gettimeofday(t_ptr, 0)
#endif

void elapsed_time(app_timer_t start, app_timer_t stop,
                  unsigned long flop);


/**********************************************************/

#endif // TIMER_H
