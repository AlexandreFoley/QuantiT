cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(QuantiT LANGUAGES CXX)

set(QUANTIT_MAJOR_VERSION 0)
set(QUANTIT_MINOR_VERSION 1)
set(QUANTIT_PATCH_VERSION 0)
set(QUANTIT_VERSION
  ${QUANTIT_MAJOR_VERSION}.${QUANTIT_MINOR_VERSION}.${QUANTIT_PATCH_VERSION})

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DEFINED ENV{EBROOTGENTOO})# we're on a compute canada supercomputer

    MESSAGE("computecanada detected")
    SET(CMAKE_SKIP_INSTALL_RPATH ON)
    SET(CMAKE_SYSTEM_PREFIX_PATH "$ENV{EBROOTGENTOO}")
    SET(PYTHON_EXECUTABLE "$ENV{EBROOTPYTHON}/bin/python")
    SET(CUDA_TOOLKIT_ROOT_DIR "$ENV{EBROOTCUDA}")
    SET(CUDA_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
    SET(CUDNN_ROOT "$ENV{EBROOTCUDNN}")
    SET(CUDNN_LIBRARY_PATH "${CUDNN_ROOT}/lib64")
    SET(CUDNN_INCLUDE_PATH "${CUDNN_ROOT}/include")
else()
    if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
        set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_TOOLKIT_ROOT_DIR}")
    else()
        message("the CUDA_TOOLKIT_ROOT_DIR environement variable isn't set, try setting it\n with the path to cuda if CMAKE or torch is having trouble finding cuda")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set(Python3_FIND_UNVERSIONED_NAMES "FIRST") 
find_package(Python3 COMPONENTS Interpreter Development)
execute_process( COMMAND which "${PYTHON_EXECUTABLE}" )
if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
  find_package(pybind11 CONFIG REQUIRED)
  set(Torch_DIR ${TORCH_INSTALL_RPATH}/../share/cmake/Torch)
else()
  set( PYTHON_EXECUTABLE python3)
  find_package(pybind11 CONFIG)
option(USE_EXTERN_TORCH "use the torch intallation located in the project extern folder" FALSE)
MESSAGE("USE EXTERNAL TORCH: ${USE_EXTERN_TORCH}")
if (USE_EXTERN_TORCH)
    set( Torch_DIR "${PROJECT_SOURCE_DIR}/extern/pytorch/torch/share/cmake/Torch")
    MESSAGE("Using Torch located in ${Torch_DIR}")
else()
    MESSAGE("Searching for Torch")
    execute_process( COMMAND "${PYTHON_EXECUTABLE}" -c "import torch;print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE Torch_DIR COMMAND_ECHO STDOUT)
    string(REGEX REPLACE "\n$" "" Torch_DIR "${Torch_DIR}" )
    set(Torch_DIR "${Torch_DIR}/Torch")
    MESSAGE("System installed Torch found: ${Torch_DIR}")
endif(USE_EXTERN_TORCH)
endif()

# Now we can find pybind11

find_package(Torch REQUIRED)
find_package(fmt 7 REQUIRED)
find_library(GOOGLE_PROFILER profiler )
if (GOOGLE_PROFILER STREQUAL "GOOGLE_PROFILER-NOTFOUND")
    set(GOOGLE_PROFILER_FOUND false)
    MESSAGE("GOOGLE PROFILER NOT FOUND")
else()
    set(GOOGLE_PROFILER_FOUND true)
endif()

if(APPLE)
    set( "-Wl,-lprofiler")
    ADD_COMPILE_DEFINITIONS(PRIVATE QTT_APPLE=true) 
else()
    set(PROFILE_NO_AS_NEEDED "-Wl,--no-as-needed,-lprofiler,--as-needed") 
endif(APPLE)

option(EMIT_PROFILER "emit profiling info in the binary" TRUE)
if(EMIT_PROFILER AND GOOGLE_PROFILER_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PROFILE_NO_AS_NEEDED}")
    add_compile_options(-fno-omit-frame-pointer)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${PROFILE_NO_AS_NEEDED}")
endif(EMIT_PROFILER AND GOOGLE_PROFILER_FOUND)
option(DISABLE_DOCTEST "disable the testing facility" FALSE)
if(DISABLE_DOCTEST)
    set(CMAKE_CXX_FLAGS "$CMAKE_CXX_FLAGS -DDOCTEST_CONFIG_DISABLE")
endif(DISABLE_DOCTEST)
option(ASAN "address sanitizer tooling" FALSE)
if( ASAN )
 add_compile_options(-fsanitize=address)
 add_link_options(-fsanitize=address)
endif(ASAN)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
link_libraries( ${TORCH_LIBRARIES} fmt::fmt-header-only )
MESSAGE("TORCH CXX LIBS: ${TORCH_CXX_FLAGS} ")

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_subdirectory(sources)
add_subdirectory(Documentation)
add_subdirectory(python_binding)

add_executable(QuantiT_dev main.cpp )
target_link_libraries(QuantiT_dev PRIVATE QuantiT)
target_compile_features(QuantiT_dev PRIVATE cxx_std_17)


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll" )
  add_custom_command(TARGET QuantiT_test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:QuantiT_test>)
endif (MSVC)