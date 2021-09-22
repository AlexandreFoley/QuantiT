



set(COMPONENT_SRCS DMRG_heisenber_test.h) #make vs-code intellisence work with with fresh, unused, header files. HEADER_LIST is definied in sources/CMakeList.txt


add_compile_options(-fno-omit-frame-pointer)

if(EMIT_PROFILER)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
    # SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif(EMIT_PROFILER)
	
set(NANB_DIR "${PROJECT_SOURCE_DIR}/extern/nanobench")
	add_executable(Benchmark benchmark.cpp)
	target_include_directories(Benchmark PRIVATE "${NANB_DIR}")
	target_compile_features(Benchmark PRIVATE cxx_std_17)
	target_link_libraries(Benchmark PRIVATE QuanTT_lib)
	add_test(NAME benchmark COMMAND Benchmark)

if(DISABLE_DOCTEST)
else()

	add_executable(QuanTT_test test.cpp)
	add_executable(QuanTT_test_lvl2 test2.cpp)

	target_compile_features(QuanTT_test PRIVATE cxx_std_17)
	target_compile_features(QuanTT_test_lvl2 PRIVATE cxx_std_17)

	target_link_libraries(QuanTT_test PRIVATE QuanTT_lib)
	target_link_libraries(QuanTT_test_lvl2 PRIVATE QuanTT_lib)

	add_test(NAME test_QuanTT COMMAND QuanTT_test)
	# add_test(NAME test_lvl2_QuanTT COMMAND QuanTT_test_lvl2) //that test takes too long to run, and require human monitoring to check correctness.
endif(DISABLE_DOCTEST)

