

set(INC_DIR "${QuanTT_SOURCE_DIR}/include")

set(HEADER_LIST "${INC_DIR}/MPT.h" "${INC_DIR}/LinearAlgebra.h" "${INC_DIR}/dimension_manip.h" "${INC_DIR}/torch_formatter.h" "${INC_DIR}/doctest.h"
                "${INC_DIR}/dmrg.h" "${INC_DIR}/operators.h" "${INC_DIR}/models.h" "${INC_DIR}/numeric.h" "${INC_DIR}/TensorGroup.h")

set(SOURCE_LIST )

# Make an automatic library - will be static or dynamic based on user setting
add_library(QuanTT_lib MPT.cpp LinearAlgebra.cpp dimension_manip.cpp torch_formatter.cpp dmrg.cpp operators.cpp 
            models.cpp TensorGroup.cpp ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(QuanTT_lib PUBLIC ../include)

# This doesn't depends on (header only) boost. present as an exemple.
# target_link_libraries(QuanTT_lib PRIVATE Boost::boost)

# All users of this library will need at least C++17
target_compile_features(QuanTT_lib PUBLIC cxx_std_17)

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})
