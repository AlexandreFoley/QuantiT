



find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules googletest during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule googletest update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update ${PROJECT_SOURCE_DIR}/extern/googletest
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/googletest/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

set(INSTALL_GTEST OFF)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

set(COMPONENT_SRCS TensorGroup_mock.h) #make vs-code intellisence work with with fresh, unused, header files. HEADER_LIST is definied in sources/CMakeList.txt

add_executable(QuanTT_mocks mocks.cpp)

target_compile_features(QuanTT_mocks PRIVATE cxx_std_17)

target_link_libraries(QuanTT_mocks PRIVATE QuanTT_lib gmock_main)

add_test(NAME test_mocks COMMAND QuanTT_mocks)