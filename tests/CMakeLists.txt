



set(COMPONENT_SRCS DMRG_heisenber_test.h) #make vs-code intellisence work with with fresh, unused, header files. HEADER_LIST is definied in sources/CMakeList.txt



add_executable(QuanTT_test test.cpp)
add_executable(QuanTT_test_lvl2 test2.cpp)

target_compile_features(QuanTT_test PRIVATE cxx_std_17)
target_compile_features(QuanTT_test_lvl2 PRIVATE cxx_std_17)

target_link_libraries(QuanTT_test PRIVATE QuanTT_lib)
target_link_libraries(QuanTT_test_lvl2 PRIVATE QuanTT_lib)

add_test(NAME test_QuanTT COMMAND QuanTT_test)
add_test(NAME test_lvl2_QuanTT COMMAND QuanTT_test_lvl2)