cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(QuantiT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(QUANTIT_MAJOR_VERSION 0)
set(QUANTIT_MINOR_VERSION 1)
set(QUANTIT_PATCH_VERSION 0)
set(QUANTIT_VERSION
    ${QUANTIT_MAJOR_VERSION}.${QUANTIT_MINOR_VERSION}.${QUANTIT_PATCH_VERSION})

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_EXTERN_TORCH
       "use the torch intallation located in the project extern folder" FALSE)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(EMIT_PROFILER "emit profiling info in the binary" TRUE)
option(DISABLE_DOCTEST "disable the testing facility" FALSE)
option(ASAN "address sanitizer tooling" FALSE)

if(DEFINED ENV{EBROOTGENTOO}) # we're on a compute canada supercomputer

  message("computecanada detected")
  # SET(CMAKE_SKIP_INSTALL_RPATH ON) #We tell cmake what to add to the rpath
  # manually. doesn't cause conflict with CC automatic rpaths
  set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{EBROOTGENTOO}")
  set(PYTHON_EXECUTABLE "$ENV{EBROOTPYTHON}/bin/python")
  set(CUDA_TOOLKIT_ROOT_DIR "$ENV{EBROOTCUDA}")
  # SET(CUDA_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
  set(CUDNN_TOOLKIT_ROOT "$ENV{EBROOTCUDNN}")
  set(CUDNN_LIBRARY_PATH "${CUDNN_TOOLKIT_ROOT}/lib64/libcudnn.so")
  set(CUDNN_INCLUDE_PATH "${CUDNN_TOOLKIT_ROOT}/include")
else()
  if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
    set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_TOOLKIT_ROOT_DIR}")
  else()
    message(
      "the CUDA_TOOLKIT_ROOT_DIR environement variable isn't set, try setting it\n with the path to cuda if CMAKE or torch is having trouble finding cuda"
    )
  endif()
endif()

set(Python3_FIND_UNVERSIONED_NAMES "FIRST") 
find_package(Python3 COMPONENTS Interpreter Numpy Development)
execute_process( COMMAND which "${PYTHON_EXECUTABLE}" )
#TODO: eliminate absolute paths to external lib and to quantit/include that appear in quantitConfig.cmake and quantitConfig-release.cmake. for partial answer see https://stackoverflow.com/questions/62000801/cmake-how-do-i-avoid-getting-absolute-paths-in-interface-target-properties
if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
  find_package(pybind11 CONFIG REQUIRED)
  set(Torch_DIR ${TORCH_INSTALL_RPATH}/../share/cmake/Torch)
else()
  set(PYTHON_EXECUTABLE python3)
  find_package(pybind11 CONFIG)
  message("USE EXTERNAL TORCH: ${USE_EXTERN_TORCH}")
  if(USE_EXTERN_TORCH)
    set(Torch_DIR
        "${PROJECT_SOURCE_DIR}/extern/pytorch/torch/share/cmake/Torch")
    message("Using Torch located in ${Torch_DIR}")
  else()
    message("Searching for Torch")
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c
              "import torch;print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE Torch_DIR)
    string(REGEX REPLACE "\n$" "" Torch_DIR "${Torch_DIR}")
    set(Torch_DIR "${Torch_DIR}/Torch")
    message("System installed Torch found: ${Torch_DIR}")
  endif(USE_EXTERN_TORCH)
  set(TORCH_INSTALL_RPATH ${Torch_DIR}/../../../lib)
endif()

# Now we can find pybind11

find_package(Torch REQUIRED)
find_package(fmt 7)
if(NOT fmt_FOUND)
  set(FMT_INSTALL ON)
  message(STATUS "fetching {fmt}")
  include(FetchContent)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY "https://github.com/fmtlib/fmt"
    GIT_TAG "8.1.1"
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(fmt)
  message(STATUS ${fmt_SOURCE_DIR})
  # add_subdirectory(${fmt_SOURCE_DIR})
endif()

find_library(GOOGLE_PROFILER profiler)
if(GOOGLE_PROFILER STREQUAL "GOOGLE_PROFILER-NOTFOUND")
  set(GOOGLE_PROFILER_FOUND false)
  set(EMIT_PROFILER false)
  message("GOOGLE PROFILER NOT FOUND")
else()
  set(GOOGLE_PROFILER_FOUND true)
endif()

if(APPLE)
  set(PROFILE_NO_AS_NEEDED "-Wl,-lprofiler")
  add_compile_definitions(PRIVATE QTT_APPLE=true)
else()
  set(PROFILE_NO_AS_NEEDED "-Wl,--no-as-needed,-lprofiler,--as-needed")
endif(APPLE)

if(EMIT_PROFILER AND GOOGLE_PROFILER_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PROFILE_NO_AS_NEEDED}")
  add_compile_options(-fno-omit-frame-pointer)
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${PROFILE_NO_AS_NEEDED}")
endif(EMIT_PROFILER AND GOOGLE_PROFILER_FOUND)
if(DISABLE_DOCTEST)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDOCTEST_CONFIG_DISABLE")
endif(DISABLE_DOCTEST)
if(ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif(ASAN)

# add_library(torch IMPORTED)
# set_target_properties(torch PROPERTIES
#   IMPORTED_LOCATION ${TORCH_LIBRARIES}
# )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
link_libraries(torch torch_library fmt::fmt-header-only)
message("TORCH CXX : ${TORCH_CXX_FLAGS} \n\n ${TORCH_LIBRARIES}")

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

add_subdirectory(sources)
add_subdirectory(Documentation)
add_subdirectory(python_binding)

add_executable(QuantiT_dev main.cpp)
target_link_libraries(QuantiT_dev PRIVATE QuantiT)
target_compile_features(QuantiT_dev PRIVATE cxx_std_17)


