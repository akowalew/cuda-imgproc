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

find_file(CONANBUILDINFO_FILE conanbuildinfo.cmake HINTS ${CMAKE_BINARY_DIR})
include(${CONANBUILDINFO_FILE})
conan_basic_setup()

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

find_package(doctest REQUIRED)

###############################################################################
# Project's targets
###############################################################################

add_subdirectory(imgproc)
add_subdirectory(process-image)
