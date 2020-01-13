###############################################################################
# CMakeLists.txt
#
# Defines build structure for `imgproc` library
#
# Author: Adam Kowalewski
# Date: 22.12.2019 13:59 CEST
###############################################################################

###############################################################################
# Library sources
###############################################################################

# Include sources common for all build versions
list(APPEND imgproc_SOURCES
    src/debug.cpp
    src/image.cpp
    src/imgproc.cpp
    src/kernel.cpp
)

if(BUILD_VERSION STREQUAL "ref")
    list(APPEND imgproc_SOURCES
        src/filter_ref.cpp
        src/hist_ref.cpp
        src/median_ref.cpp
    )
elseif(BUILD_VERSION STREQUAL "cpu_single")
    list(APPEND imgproc_SOURCES
        src/filter_cpu_single.cpp
        src/hist_cpu_single.cpp
        src/median_cpu_single.cpp
    )
elseif(BUILD_VERSION STREQUAL "cpu_multi")
    list(APPEND imgproc_SOURCES
        src/filter_cpu_multi.cpp
        src/hist_cpu_multi.cpp
        src/median_cpu_multi.cpp
    )
elseif(BUILD_VERSION STREQUAL "cuda_v1")
    list(APPEND imgproc_SOURCES
        src/filter_cuda_v1.cu
        src/hist_cuda_v1.cu
        src/image_cuda.cpp
        src/imgproc_cuda.cpp
        src/median_cuda_v1.cu
    )
elseif(BUILD_VERSION STREQUAL "cuda_v2")
    list(APPEND imgproc_SOURCES
        src/filter_cuda_v2.cu
        src/hist_cuda_v2.cu
        src/image_cuda.cpp
        src/imgproc_cuda.cpp
        src/median_cuda_v2.cu
    )
else()
    message(FATAL_ERROR "Wrong BUILD_VERSION=${BUILD_VERSION}")
endif()

###############################################################################
# Library definition
###############################################################################

add_library(imgproc
    ${imgproc_SOURCES}
)

target_include_directories(imgproc
    PUBLIC
        include
)

target_link_libraries(imgproc
	PUBLIC
		${OpenCV_LIBRARIES}
        cuda-samples
)

if(BUILD_VERSION STREQUAL "cpu_multi" AND OpenMP_CXX_FOUND)
    # If OpenMP is needed to build cpu_multi, link against it
    target_link_libraries(imgproc PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(imgproc
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

if(GNU)
    target_compile_options(imgproc
        PRIVATE
            # Enable all GCC warnings
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>
    )
endif()

if(BUILD_VERSION STREQUAL "cuda_v1" OR BUILD_VERSION STREQUAL "cuda_v2")
    target_compile_options(imgproc
        PRIVATE
            # Allow to use constexpr functions in CUDA kernels
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

###############################################################################
# Benchmarks
###############################################################################

if(BUILD_BENCHMARKING)
    add_executable(imgproc-bench
        bench/bench_main.cpp
        bench/hist_${BUILD_VERSION}_bench.cpp
    )

    target_link_libraries(imgproc-bench
        PRIVATE
            imgproc
            -fopenmp

            ${benchmark_LIBRARIES}
            benchmark
            pthread
    )

    if(GNU)
        target_compile_options(imgproc-bench
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
    add_executable(imgproc-test
	    test/test_main.cpp

        test/hist_cuda1_test.cpp
    )

    target_link_libraries(imgproc-test
	    PRIVATE
		    imgproc
            -fopenmp
            doctest::doctest
    )

    set_target_properties(imgproc-test
        PROPERTIES
            CXX_STANDARD 17
    )

    if(GNU)
        target_compile_options(imgproc-test
            PRIVATE
                # Enable all GCC warnings
                -Wall -Wextra -pedantic
        )
    endif()
endif()
