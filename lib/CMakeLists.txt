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
    src/debug.cpp
    src/image.cpp
    src/kernel.cpp
    src/proc.cpp
    src/reader.cpp
    src/writer.cpp
)

if(BUILD_VERSION STREQUAL CUDA)
    list(APPEND imgproc_SOURCES
        src/cuda/cuda_filter.cu
        src/cuda/cuda_hist.cu
        src/cuda/cuda_host_image.cu
        src/cuda/cuda_image.cu
        src/cuda/cuda_kernel.cu
        src/cuda/cuda_lut.cu
        src/cuda/cuda_median.cu
        src/cuda/cuda_proc.cu
    )
else()
    list(APPEND imgproc_SOURCES
        src/cpu/cpu_filter.cpp
        src/cpu/cpu_hist.cpp
        src/cpu/cpu_median.cpp
        src/cpu/cpu_proc.cpp
    )
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

    PRIVATE
        src/
        src/cuda
        src/cpu
)

target_compile_definitions(imgproc
    PRIVATE
        -DBUILD_VERSION_${BUILD_VERSION}
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
    )

    if(BUILD_VERSION STREQUAL CUDA)
        list(APPEND imgproc-bench_SOURCES
            bench/cuda/cuda_filter_bench.cu
            bench/cuda/cuda_hist_bench.cu
            bench/cuda/cuda_lut_bench.cu
            bench/cuda/cuda_median_bench.cu
        )
    endif()

    add_executable(imgproc-bench
        ${imgproc-bench_SOURCES}
    )

    target_link_libraries(imgproc-bench
        PRIVATE
            imgproc

            benchmark::benchmark
    )

    target_include_directories(imgproc-bench
        PRIVATE
            $<TARGET_PROPERTY:imgproc,INCLUDE_DIRECTORIES>
            bench/cuda
            bench
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
    if(BUILD_VERSION STREQUAL CUDA)
        list(APPEND imgproc-test_SOURCES
            test/cuda_hist_test.cu
            test/cuda_lut_test.cu
        )
    endif()

    add_executable(imgproc-test
	    test/test_main.cpp
        ${imgproc-test_SOURCES}
    )
    target_include_directories(imgproc-test
        PRIVATE
            $<TARGET_PROPERTY:imgproc,INCLUDE_DIRECTORIES>
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
