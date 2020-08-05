


set(HEADER_LIST "${QuanTT_SOURCE_DIR}/include/MPT.h" "${QuanTT_SOURCE_DIR}/include/LinearAlgebra.h")

# Make an automatic library - will be static or dynamic based on user setting
add_library(QuanTT_lib MPT.cpp LinearAlgebra.cpp ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(QuanTT_lib PUBLIC ../include)

# This doesn't depends on (header only) boost. present as an exemple.
# target_link_libraries(QuanTT_lib PRIVATE Boost::boost)

# All users of this library will need at least C++17
target_compile_features(QuanTT_lib PUBLIC cxx_std_17)

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})
