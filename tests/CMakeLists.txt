



add_executable(QuanTT_test test.cpp)

target_compile_features(QuanTT_test PRIVATE cxx_std_17)

target_link_libraries(QuanTT_test PRIVATE QuanTT_lib)

add_test(NAME test_QuanTT COMMAND QuanTT_test)