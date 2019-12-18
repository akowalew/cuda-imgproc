###############################################################################
# CMakeLists.txt
#
# Defines project structure, build options and its targets
#
# Author: Adam Kowalewski
# Date: 4.11.2019 18:06 CEST
###############################################################################

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(cuda-imgproc VERSION 0.1.0 LANGUAGES CXX)

###############################################################################
# Global options
###############################################################################

option(BUILD_TESTING "Whether to build tests or not" ON)
option(BUILD_DEBUG "Whether to print debug information" ON)
option(BUILD_BENCHMARKING "Whether to build benchmarks or not" ON)
option(BUILD_CUDA "Whether to build CUDA-related code" ON)
option(BUILD_CONAN "Whether to use Conan package manager or not" OFF)

set(BUILD_VERSION "reference" CACHE STRING "Defines function implementation version")
set_property(CACHE BUILD_VERSION PROPERTY STRINGS reference cpuserial cpuparallel cuda1 cuda2)

###############################################################################
# Project global options
###############################################################################

if(BUILD_CUDA)
    # If needed, enable CUDA language compilers
    enable_language(CUDA)
endif()

if(BUILD_TESTING)
    # If unit-tests needed, enable CMake testing suite
	enable_testing()
endif()

if(BUILD_DEBUG)
    add_definitions(-DDEBUG)
endif()

###############################################################################
# Conan integration
###############################################################################

# At the moment, Conan is useful only when building tests
if(BUILD_CONAN)
    # Include output of conan installation
    find_file(CONANBUILDINFO_FILE conanbuildinfo.cmake HINTS ${CMAKE_BINARY_DIR})
    include(${CONANBUILDINFO_FILE})

    # Perform basic Conan setup:
    #  - read installed dependencies
    #  - configure output directories (bin, lib)
    #  - adjust compiler flags
    conan_basic_setup()
endif()

###############################################################################
# External packages
###############################################################################

# We need OpenCV solely for image reading-writing and image API
find_package(OpenCV REQUIRED
    COMPONENTS
    	opencv_core
    	opencv_imgproc
    	opencv_videoio
    	opencv_highgui
    	opencv_imgcodecs
)

if(BUILD_BENCHMARKING)
    # We need this library only when benchmarking
    find_package(benchmark REQUIRED)
endif()

if(BUILD_TESTING)
	# We need this framework only when testing
	find_package(doctest REQUIRED)
endif()

if(BUILD_VERSION STREQUAL "cpuparallel")
    # We need OpenMP extensions when building for CPU
    find_package(OpenMP)
endif()

###############################################################################
# Common object files
###############################################################################

file(GLOB hist_SOURCES src/hist_${BUILD_VERSION}.c*)
file(GLOB filter_SOURCES src/filter_${BUILD_VERSION}.c*)

# Define library contained only from compiled object files
add_library(common OBJECT
	src/core.cpp
	src/debug.cpp
	src/median.cpp
	src/processor.cpp
	src/reader.cpp
	src/writer.cpp
    src/timer.cpp

    ${hist_SOURCES}
    ${filter_SOURCES}
)

target_include_directories(common
    PUBLIC
        include
)

target_link_libraries(common
	PUBLIC
		${OpenCV_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    # If OpenMP is needed, link against it
    target_link_libraries(common PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(common
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

if(GNU)
    target_compile_options(common
        PRIVATE
            # Enable all GCC warnings
            -Wall -Wextra -pedantic
    )
endif()

###############################################################################
# Main executable
###############################################################################

add_executable(process-image
	src/main.cpp
)

target_link_libraries(process-image
	PRIVATE
		common
)

if(GNU)
    target_compile_options(process-image
        PRIVATE
            # Enable all GCC warnings
            -Wall -Wextra -pedantic
    )
endif()

###############################################################################
# Benchmarks
###############################################################################

if(BUILD_BENCHMARKING)
    add_executable(process-image-bench
        bench/bench_main.cpp
        bench/filter_bench.cpp
        bench/hist_bench.cpp
        bench/median_bench.cpp
    )

    target_link_libraries(process-image-bench
        PRIVATE
            common

            ${benchmark_LIBRARIES}
            benchmark
            pthread
    )

    if(GNU)
        target_compile_options(process-image-bench
            PRIVATE
                # Enable all GCC warnings
                -Wall -Wextra -pedantic
        )
    endif()
endif()

###############################################################################
# Unit tests
###############################################################################

if(BUILD_TESTING)
    add_executable(process-image-test
	    test/test_main.cpp
    )

    target_link_libraries(process-image-test
	    PRIVATE
		    common

            doctest::doctest
    )

    if(GNU)
        target_compile_options(process-image-test
            PRIVATE
                # Enable all GCC warnings
                -Wall -Wextra -pedantic
        )
    endif()
endif()
