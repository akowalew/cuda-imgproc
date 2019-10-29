cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(cuda-imgproc VERSION 0.1.0 LANGUAGES CXX)

add_executable(cuda-imgproc
    src/main.cpp
)

target_compile_options(cuda-imgproc
    PRIVATE
        -Wall -Wextra -pedantic
)

set_target_properties(cuda-imgproc
    PROPERTIES
        CXX_STANDARD 14
)
