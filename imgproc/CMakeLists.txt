###############################################################################
# CMakeLists.txt
#
# Defines build structure for imgproc library
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 4.11.2019 18:21 CEST
###############################################################################

add_library(imgproc
    src/format.cpp
    src/imgproc.cpp
)

target_include_directories(imgproc
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(imgproc
    PRIVATE
        ${OpenCV_LIBS}
)

target_compile_options(imgproc
    PRIVATE
        -Wall -Wextra -pedantic
)

set_target_properties(imgproc
    PROPERTIES
        CXX_STANDARD 14
)

target_compile_options(imgproc
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>
)
