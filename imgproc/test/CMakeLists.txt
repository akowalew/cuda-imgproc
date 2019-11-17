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

add_executable(imgproc_test
    test_main.cpp
)

target_link_libraries(imgproc_test
    PRIVATE
        imgproc
        doctest::doctest
)

set_target_properties(imgproc_test
    PROPERTIES
        CXX_STANDARD 14
)

target_compile_options(imgproc_test
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>
)

###############################################################################
# Test definition
###############################################################################

add_test(imgproc_test
    ${CMAKE_BINARY_DIR}/bin/imgproc_test
)
