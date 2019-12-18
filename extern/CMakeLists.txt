###############################################################################
# CMakeLists.txt
# 
# Configures extern targets (like submodules) with project build
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 18.12.2019 3:24
###############################################################################

add_library(cuda-samples INTERFACE)

target_include_directories(cuda-samples
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda-samples/Common
)
