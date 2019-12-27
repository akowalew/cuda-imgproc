###############################################################################
# CMakeLists.txt
#
# Defines build structure for `imgproc` library
#
# Author: Adam Kowalewski
# Date: 22.12.2019 13:59 CEST
###############################################################################

set(BUILD_VERSION "reference" CACHE STRING "Defines function implementation version")
set_property(CACHE BUILD_VERSION PROPERTY STRINGS ref cpu_single cpu_multi cuda_v1 cuda_v2)

###############################################################################
# Library sources
###############################################################################

list(APPEND imgproc_SOURCES
    src/filter_${BUILD_VERSION}
    src/median_${BUILD_VERSION}
    src/hist_${BUILD_VERSION}
)

if(BUILD_VERSION STREQUAL "cuda_v1" OR BUILD_VERSION STREQUAL "cuda_v2")
    list(APPEND imgproc_SOURCES
        src/image_cuda.cpp
        src/imgproc_cuda.cpp
    )
else()
    list(APPEND imgproc_SOURCES
        src/imgproc.cpp
    )
endif()

###############################################################################
# Library definition
###############################################################################

add_library(imgproc OBJECT
	src/debug.cpp

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

        # test/hist_cuda1_test.cpp
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
