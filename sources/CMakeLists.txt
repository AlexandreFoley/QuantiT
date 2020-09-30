

set(INC_DIR "${QuanTT_SOURCE_DIR}/include")
set(BST_DIR "${QuanTT_SOURCE_DIR}/include/boost")
set(GRP_DIR "${INC_DIR}/Groups")
set(DCT_DIR "${INC_DIR}/doctest")

set(HEADER_LIST "${INC_DIR}/MPT.h" "${INC_DIR}/LinearAlgebra.h" "${INC_DIR}/dimension_manip.h" 
                "${INC_DIR}/torch_formatter.h" "${DCT_DIR}/doctest.h" "${DCT_DIR}/doctest_redef.h" 
                "${DCT_DIR}/cond_doctest.h" "${INC_DIR}/dmrg.h" "${INC_DIR}/operators.h" 
                "${INC_DIR}/models.h" "${INC_DIR}/numeric.h" "${INC_DIR}/templateMeta.h" 
                "${BST_DIR}/stl_interfaces/config.hpp" "${BST_DIR}/stl_interfaces/fwd.hpp"
                "${BST_DIR}/stl_interfaces/iterator_interface.hpp" "${BST_DIR}/stl_interfaces/view_interface.hpp"
                "${BST_DIR}/stl_interfaces/sequence_container_interface.hpp" "${GRP_DIR}/cgroup_container.h"
                "${GRP_DIR}/cgroup_impl.h" "${GRP_DIR}/composite_group.h" "${GRP_DIR}/cgroup_container_impl.h"
                "${GRP_DIR}/groups.h" "${GRP_DIR}/groups_utils.h")

set(SOURCE_LIST )

# Make an automatic library - will be static or dynamic based on user setting
add_library(QuanTT_lib MPT.cpp LinearAlgebra.cpp dimension_manip.cpp torch_formatter.cpp dmrg.cpp operators.cpp 
            models.cpp composite_group.cpp groups.cpp ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(QuanTT_lib PUBLIC ../include)

# This doesn't depends on (header only) boost. present as an exemple.
# target_link_libraries(QuanTT_lib PRIVATE Boost::boost)

# All users of this library will need at least C++17
target_compile_features(QuanTT_lib PUBLIC cxx_std_17)

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})
