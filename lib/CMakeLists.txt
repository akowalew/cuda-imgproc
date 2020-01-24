###############################################################################
# CMakeLists.txt
#
# Defines build structure for `imgproc` library
###############################################################################

###############################################################################
# Library sources
###############################################################################

# Include sources common for all build versions
list(APPEND imgproc_SOURCES
    src/cuda/cuda_filter.cu
    src/cuda/cuda_hist.cu
    src/cuda/cuda_image.cu
    src/cuda/cuda_median.cu
    src/cuda/cuda_proc.cu

    src/debug.cpp
    src/image.cpp
    src/proc.cpp
    src/reader.cpp
    src/writer.cpp
)

###############################################################################
# Library definition
###############################################################################

add_library(imgproc
    ${imgproc_SOURCES}
)

target_include_directories(imgproc
    PUBLIC
        include

    PRIVATE
        src/
        src/cuda
)

target_link_libraries(imgproc
	PUBLIC
		${OpenCV_LIBRARIES}
        cuda-samples
)

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

target_compile_options(imgproc
    PRIVATE
        # Allow to use constexpr functions in CUDA kernels
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

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
