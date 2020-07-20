cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(QuanTT)

set( Torch_DIR /home/alex/projects/QuanTT/pytorch_dbg/torch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)

include(CTest)
enable_testing()


add_executable(QuanTT_test main.cpp)
target_link_libraries(QuanTT_test "${TORCH_LIBRARIES}" fmt)
set_property(TARGET QuanTT_test PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET QuanTT_test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:QuanTT_test>)
endif (MSVC)