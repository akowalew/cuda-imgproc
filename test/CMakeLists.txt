###############################################################################
# CMakeLists.txt
#
# Defines build structure of tests for imgproc library
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 17.11.2019 16:27 CEST
###############################################################################

###############################################################################
# Test executable definition
###############################################################################

add_executable(imgproc-test
    test_main.cpp
)

target_link_libraries(imgproc-test
    PRIVATE
        imgproc
)

set_target_properties(imgproc-test
    PROPERTIES
        CXX_STANDARD 14
)

target_compile_options(imgproc-test
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>
)

###############################################################################
# Test definition
###############################################################################

add_test(imgproc-test
    ${CMAKE_BINARY_DIR}/bin/imgproc-test
)
