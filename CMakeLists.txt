###############################################################################
# CMakeLists.txt
#
# Defines project structure, build options and its targets
###############################################################################

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(cuda-imgproc VERSION 0.1.0 LANGUAGES CXX CUDA)

###############################################################################
# Global options
###############################################################################

option(BUILD_APP "Whether to build process-image app or not" ON)
option(BUILD_LIB "Whether to build imgproc lib or not" ON)
option(BUILD_TESTING "Whether to build tests or not" ON)
option(BUILD_DEBUG "Whether to print debug information" ON)
option(BUILD_BENCHMARKING "Whether to build benchmarks or not" ON)
option(BUILD_CONAN "Whether to use Conan package manager or not" ON)
option(BUILD_EXTERN "Whether to build extern libs or not" ON)
option(BUILD_VERBOSE "Whether to enable build verbosity or not" ON)

set(LOG_LEVEL "OFF" CACHE STRING "Sets desired log level across the project")
set_property(CACHE LOG_LEVEL PROPERTY STRINGS TRACE DEBUG INFO WARNING ERROR OFF)

###############################################################################
# Project global options
###############################################################################

if(BUILD_TESTING)
    # If unit-tests needed, enable CMake testing suite
	enable_testing()
endif()

if(BUILD_DEBUG)
    add_definitions(-DDEBUG)
endif()

# Set desired log level across the project
add_definitions(-DLOG_LEVEL=LOG_LEVEL_${LOG_LEVEL})

# If some CXX or CUDA standard is needed for a target, require it
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# If verbose, print additional messages during build
if(BUILD_VERBOSE)
    # When building CUDA code, print registers usage
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )
endif()

###############################################################################
# Extern libraries
###############################################################################

if(BUILD_EXTERN)
    add_subdirectory(extern)
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

# Ensure that we are using old cxx ABI (required for OpenCV...)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

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
    find_package(benchmark CONFIG REQUIRED)
endif()

if(BUILD_TESTING)
	# We need this framework only when testing
	find_package(doctest REQUIRED)
endif()

if(BUILD_VERSION STREQUAL "cpu_multi")
    # We need OpenMP extensions when building for CPU with multicore configuration
    find_package(OpenMP REQUIRED)
endif()

###############################################################################
# Subprojects
###############################################################################

add_subdirectory(lib)
add_subdirectory(app)
