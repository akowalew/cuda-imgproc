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
elseif(BUILD_VERSION STREQUAL "cuda")
    list(APPEND imgproc_SOURCES
        src/filter_cuda.cu
        src/hist_cuda.cu
        src/image_cuda.cpp
        src/imgproc_cuda.cpp
        src/median_cuda.cu
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

if(BUILD_VERSION STREQUAL "cuda")
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
    list(APPEND imgproc-bench_SOURCES
        bench/bench_main.cpp
        bench/filter_bench.cpp
        bench/hist_bench.cpp
        bench/median_bench.cpp
    )

    if(BUILD_VERSION STREQUAL "ref")

    elseif(BUILD_VERSION STREQUAL "cpu_single")

    elseif(BUILD_VERSION STREQUAL "cpu_multi")

    elseif(BUILD_VERSION STREQUAL "cuda")
        list(APPEND imgproc-bench_SOURCES
            bench/hist_cuda_bench.cpp
        )
    else()
        message(FATAL_ERROR "Wrong BUILD_VERSION=${BUILD_VERSION}")
    endif()

    add_executable(imgproc-bench
        ${imgproc-bench_SOURCES}
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
