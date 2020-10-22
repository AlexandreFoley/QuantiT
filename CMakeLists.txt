cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(QuanTT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set( Torch_DIR "${QuanTT_SOURCE_DIR}/extern/pytorch/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(fmt 6 REQUIRED)

link_libraries( ${TORCH_LIBRARIES} fmt)



if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_subdirectory(sources)

add_executable(QuanTT_dev main.cpp )
target_link_libraries(QuanTT_dev PRIVATE QuanTT_lib)
target_compile_features(QuanTT_dev PRIVATE cxx_std_17)

set(COMPONENT_SRCS ${HEADER_LIST}) #make vs-code intellisence work with with fresh, unused, header files. HEADER_LIST is definied in sources/CMakeList.txt

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll" )
  add_custom_command(TARGET QuanTT_test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:QuanTT_test>)
endif (MSVC)