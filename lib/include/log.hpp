///////////////////////////////////////////////////////////////////////////////
// log.cuh
//
// Contains declarations of debug facilities for CUDA
///////////////////////////////////////////////////////////////////////////////

#pragma once

#define LOG_LEVEL_TRACE 0x1f
#define LOG_LEVEL_DEBUG 0x0f
#define LOG_LEVEL_INFO 0x07
#define LOG_LEVEL_WARNING 0x03
#define LOG_LEVEL_ERROR 0x01
#define LOG_LEVEL_OFF 0x00

#ifndef LOG_LEVEL
# define LOG_LEVEL LOG_LEVEL_OFF
#endif

#if LOG_LEVEL != LOG_LEVEL_OFF
# include <cstdio>
#endif

#if LOG_LEVEL & LOG_LEVEL_TRACE 
# define LOG_TRACE(format, ...) printf("[TRACE] " format, __VA_ARGS__);
#else
# define LOG_TRACE(format, ...) 
#endif

#if LOG_LEVEL & LOG_LEVEL_DEBUG 
# define LOG_DEBUG(format, ...) printf("[DEBUG] " format, __VA_ARGS__);
#else
# define LOG_DEBUG(format, ...) 
#endif

#if LOG_LEVEL & LOG_LEVEL_INFO 
# define LOG_INFO(format, ...) printf("[INFO] " format, ##__VA_ARGS__);
#else
# define LOG_INFO(format, ...) 
#endif

#if LOG_LEVEL & LOG_LEVEL_WARNING 
# define LOG_WARNING(format, ...) printf("[WARNING] " format, __VA_ARGS__);
#else
# define LOG_WARNING(format, ...) 
#endif

#if LOG_LEVEL & LOG_LEVEL_ERROR 
# define LOG_ERROR(format, ...) printf("[ERROR] " format, __VA_ARGS__);
#else
# define LOG_ERROR(format, ...) 
#endif

