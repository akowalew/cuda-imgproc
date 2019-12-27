###############################################################################
# CMakeLists.txt
#
# Defines build structure for `process-image` application
#
# Author: Adam Kowalewski
# Date: 22.12.2019 13:57 CEST
###############################################################################

###############################################################################
# Executable definition
###############################################################################

add_executable(process-image
	src/main.cpp
)

target_link_libraries(process-image
	PRIVATE
		imgproc
)

set_target_properties(process-image
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

if(GNU)
    target_compile_options(process-image
        PRIVATE
            # Enable all GCC warnings
            -Wall -Wextra -pedantic
    )
endif()