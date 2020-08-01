cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(QuanTT)

set(CMAKE_CXX_STANDARD 17)
set( Torch_DIR "${QuanTT_SOURCE_DIR}/extern/pytorch/torch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)

include_directories(include)

link_libraries( ${TORCH_LIBRARIES} fmt)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_subdirectory(sources)
add_executable(QuanTT_dev main.cpp)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll" )
  add_custom_command(TARGET QuanTT_test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:QuanTT_test>)
endif (MSVC)