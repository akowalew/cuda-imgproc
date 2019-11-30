###############################################################################
# CMakeLists.txt
#
# Defines project structure, build options and its targets
#
# Author: Adam Kowalewski
# Date: 4.11.2019 18:06 CEST
###############################################################################

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(cuda-imgproc VERSION 0.1.0 LANGUAGES CXX #[[CUDA]])

###############################################################################
# Global options
###############################################################################

option(BUILD_TESTING "Whether to build tests or not" ON)

###############################################################################
# Project global options
###############################################################################

if(BUILD_TESTING)
	enable_testing()
endif()

###############################################################################
# Conan integration
###############################################################################

# At the moment, Conan is useful only when building tests
if(BUILD_TESTING)
    # Include output of conan installation
    find_file(CONANBUILDINFO_FILE conanbuildinfo.cmake HINTS ${CMAKE_BINARY_DIR})
    include(${CONANBUILDINFO_FILE})
    conan_basic_setup()
endif()

###############################################################################
# External packages
###############################################################################

find_package(OpenCV REQUIRED
    COMPONENTS
    	opencv_core
    	opencv_imgproc
    	opencv_videoio
    	opencv_highgui
    	opencv_imgcodecs
)

if(BUILD_TESTING)
	# We need this framework only when testing
	find_package(doctest REQUIRED)
endif()

###############################################################################
# Global options
###############################################################################

# Set global warning levels
if(GNU)
	add_compile_options(
		-Wall -Wextra -pedantic
	)
endif()

# Set global include directory
include_directories(include)

# Require C++17 standard without platform extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# Common object files
###############################################################################

# Define quasi-library contained only from object files
add_library(common OBJECT
	src/core.cpp
	src/debug.cpp
	src/filter.cpp
	src/hist.cpp
	src/median.cpp
	src/processor.cpp
	src/reader.cpp
	src/writer.cpp
  src/timer.cpp
)

target_link_libraries(common
	PUBLIC
		${OpenCV_LIBRARIES}
)

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
endif()
