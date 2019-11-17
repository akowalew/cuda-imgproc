###############################################################################
# CMakeLists.txt
#
# Defines build structure for `process-image` application
#
# Author: Adam Kowalewski (ram.techen@gmail.com)
# Date: 4.11.2019 18:18 CEST
###############################################################################

add_executable(process-image
    src/main.cpp

    src/ImageProcessor.cpp
)

target_compile_options(process-image
    PRIVATE
        -Wall -Wextra -pedantic
)

set_target_properties(process-image
    PROPERTIES
        CXX_STANDARD 14
)

target_compile_options(process-image
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>
)

target_link_libraries(process-image
    PRIVATE
        imgproc
        ${OpenCV_LIBS}
)

target_include_directories(process-image
    PRIVATE
        include
        src
)
