




# if(EMIT_PROFILER)
#     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#     SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#     SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
#     # SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
# endif(EMIT_PROFILER)

add_executable(heisenberg2D 2Dheisenberg.cpp )
target_compile_features(heisenberg2D PRIVATE cxx_std_20)
target_link_libraries(heisenberg2D PRIVATE QuantiT)

set(NANB_DIR "${PROJECT_SOURCE_DIR}/extern/nanobench")
	add_executable(Benchmark benchmark.cpp)
	target_include_directories(Benchmark PRIVATE "${NANB_DIR}")
	target_compile_features(Benchmark PRIVATE cxx_std_17)
	target_link_libraries(Benchmark PRIVATE QuantiT)
	if(EMIT_PROFILER)
	target_compile_definitions(Benchmark PRIVATE E_PROFILER=true)
	endif(EMIT_PROFILER)
	add_test(NAME benchmark COMMAND Benchmark)

if(DISABLE_DOCTEST)
else()

	add_executable(QuantiT_test test.cpp)
	add_executable(QuantiT_test_lvl2 test2.cpp)

	target_compile_features(QuantiT_test PRIVATE cxx_std_17)
	target_compile_features(QuantiT_test_lvl2 PRIVATE cxx_std_17)

	target_link_libraries(QuantiT_test PRIVATE QuantiT)
	target_link_libraries(QuantiT_test_lvl2 PRIVATE QuantiT)

	add_test(NAME test_QuantiT COMMAND QuantiT_test)
	# add_test(NAME test_lvl2_QuantiT COMMAND QuantiT_test_lvl2) //that test takes too long to run, and require human monitoring to check correctness.
endif(DISABLE_DOCTEST)

