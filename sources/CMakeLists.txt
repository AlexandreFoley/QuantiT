

set(INC_DIR "${PROJECT_SOURCE_DIR}/include")
set(BST_DIR "${PROJECT_SOURCE_DIR}/extern/boost/boost")
set(GRP_DIR "${INC_DIR}/Conserved")
set(DCT_DIR "${PROJECT_SOURCE_DIR}/extern/doctest")
set(BTEN_DIR "${INC_DIR}/blockTensor")

set(HEADER_LIST "${INC_DIR}/MPT.h" "${INC_DIR}/LinearAlgebra.h" "${INC_DIR}/dimension_manip.h" 
                "${INC_DIR}/torch_formatter.h" "${DCT_DIR}/doctest.h" "${INC_DIR}/doctest/doctest_proxy.h" 
                "${INC_DIR}/dmrg.h" "${INC_DIR}/operators.h" 
                "${INC_DIR}/models.h" "${INC_DIR}/numeric.h" "${INC_DIR}/templateMeta.h" 
                "${BST_DIR}/stl_interfaces/config.hpp" "${BST_DIR}/stl_interfaces/fwd.hpp"
                "${BST_DIR}/stl_interfaces/iterator_interface.hpp" "${BST_DIR}/stl_interfaces/view_interface.hpp"
                "${BST_DIR}/stl_interfaces/sequence_container_interface.hpp" "${GRP_DIR}/Composite/quantity_vector.h"
                "${GRP_DIR}/Composite/quantity_impl.h" "${GRP_DIR}/Composite/cquantity.h" "${GRP_DIR}/Composite/quantity_vector_impl.h"
                "${GRP_DIR}/quantity.h" "${GRP_DIR}/quantity_utils.h" "${BTEN_DIR}/btensor.h" "${BTEN_DIR}/flat_map.h"
                "${INC_DIR}/tensorgdot.h" "${BTEN_DIR}/LinearAlgebra.h" "${INC_DIR}/dmrg_logger.h" "${INC_DIR/dmrg_options.h}")

set(SOURCE_LIST MPT.cpp LinearAlgebra.cpp dimension_manip.cpp torch_formatter.cpp dmrg.cpp operators.cpp 
models.cpp any_quantity.cpp groups.cpp btensor.cpp tensorgdot.cpp btensor_linalg.cpp dmrg_logger.cpp )

# Make an automatic library - will be static or dynamic based on user setting
add_library(QuantiT ${SOURCE_LIST} ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(QuantiT PUBLIC ${CMAKE_SOURCE_DIR}/extern/boost ${CMAKE_SOURCE_DIR}/extern/nanobench ${CMAKE_SOURCE_DIR}/extern/doctest ${CMAKE_SOURCE_DIR}/include)

# This doesn't depends on (header only) boost. present as an exemple.
# target_link_libraries(QuantiT_lib PRIVATE Boost::boost)

# All users of this library will need at least C++17
target_compile_features(QuantiT PUBLIC cxx_std_17)

if(SKBUILD)
   install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/"  TYPE INCLUDE
   COMPONENT QuantiT)
   install(DIRECTORY "${CMAKE_SOURCE_DIR}/extern/doctest" TYPE INCLUDE
   COMPONENT QuantiT)
   install(DIRECTORY "${CMAKE_SOURCE_DIR}/extern/boost" TYPE INCLUDE
   COMPONENT QuantiT)
   install(DIRECTORY "${CMAKE_SOURCE_DIR}/extern/nanobench" TYPE INCLUDE
   COMPONENT QuantiT)
   MESSAGE("INSTALL PREFIX"p ${CMAKE_INSTALL_PREFIX})
#    target_include_directories(QuantiT PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/extern/boost ${CMAKE_SOURCE_DIR}/extern/doctest ${CMAKE_SOURCE_DIR}/include>  $<INSTALL_INTERFACE:extern/boost extern/doctest include >)
   install(TARGETS QuantiT DESTINATION ./lib EXPORT quantitConfig LIBRARY DESTINATION ./lib  ARCHIVE DESTINATION ./lib  INCLUDES DESTINATION ./include/ )
   set_target_properties(QuantiT PROPERTIES INSTALL_RPATH ${TORCH_INSTALL_RPATH})
#    set_target_properties(QuantiT PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "./include ./extern/doctest ./extern/boost ./extern/nanobench")
#    install(TARGETS
#         your_target
#     EXPORT YourPackageConfig
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )
    export(TARGETS
            QuantiT
        NAMESPACE QuantiT::
        FILE "./quantitConfig.cmake"
    )
#     install(EXPORT
#     quantitConfig
# DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake"
# NAMESPACE quantit::
# )

endif()