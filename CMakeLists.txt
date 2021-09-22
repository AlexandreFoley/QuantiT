cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(QuantiT LANGUAGES CXX)


if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
    set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_TOOLKIT_ROOT_DIR}")
else()
    message("the CUDA_TOOLKIT_ROOT_DIR environement variable isn't set, try setting it\n with the path to cuda if CMAKE or torch is having trouble finding cuda")
endif()


set(CMAKE_CXX_STANDARD 17)
if(APPLE)
endif(APPLE)
find_package(Python3 COMPONENTS Interpreter Development)
option(USE_EXTERN_TORCH "use the torch intallation located in the project extern folder" FALSE)
MESSAGE("USE EXTERNAL TORCH: ${USE_EXTERN_TORCH}")
if (USE_EXTERN_TORCH)
    set( Torch_DIR "${PROJECT_SOURCE_DIR}/extern/pytorch/torch/share/cmake/Torch")
    MESSAGE("Using Torch located in ${Torch_DIR}")
else()
    MESSAGE("Searching for Torch")
    execute_process( COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE Torch_DIR)
    string(REGEX REPLACE "\n$" "" Torch_DIR "${Torch_DIR}" )
    set(Torch_DIR "${Torch_DIR}/Torch")
    MESSAGE("System installed Torch found: ${Torch_DIR}")
endif(USE_EXTERN_TORCH)
find_package(Torch REQUIRED)
find_package(fmt 7 REQUIRED)

option(EMIT_PROFILER "emit profiling info in the binary" FALSE)
if(EMIT_PROFILER)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
    # SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif(EMIT_PROFILER)
option(DISABLE_DOCTEST "disable the testing facility" FALSE)
if(DISABLE_DOCTEST)
    set(CMAKE_CXX_FLAGS "$CMAKE_CXX_FLAGS -DDOCTEST_CONFIG_DISABLE")
endif(DISABLE_DOCTEST)

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

link_libraries( ${TORCH_LIBRARIES} fmt::fmt-header-only )

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_subdirectory(sources)
add_subdirectory(Documentation)

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