



if(pybind11_FOUND)

  MESSAGE("PYTORCH LIB" ${TORCH_INSTALL_PREFIX})
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
  MESSAGE("TORCH LIB: " ${TORCH_PYTHON_LIBRARY} ${PYTHON_TORCH_LIBRARY})

  add_library(QuantiT MODULE quantt.cpp conserved.cpp linalg.cpp)
  
  target_link_libraries(QuantiT PUBLIC pybind11::module ${TORCH_PYTHON_LIBRARY} QuanTT_lib)
  set_target_properties( QuantiT PROPERTIES PREFIX "") #remove the "lib" from the resulting shared object

endif(pybind11_FOUND)