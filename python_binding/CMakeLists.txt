



if(pybind11_FOUND)

  MESSAGE("PYTORCH LIB" ${TORCH_INSTALL_PREFIX})
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
  MESSAGE("TORCH LIB: " ${TORCH_PYTHON_LIBRARY} ${PYTHON_TORCH_LIBRARY})


  add_library(quantit MODULE quantt.cpp conserved.cpp linalg.cpp networks.cpp operators.cpp algorithm.cpp python_file)
  target_link_libraries(quantit PUBLIC pybind11::module ${TORCH_PYTHON_LIBRARY} QuanTT_lib)
  set_target_properties( quantit PROPERTIES PREFIX "") #remove the "lib" from the resulting shared object

  set(EX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/exemples)
  set(PYTHON_EXEMPLES EX_DIR/dmrg.py)

  add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/python_binding/python_file
                    COMMAND ${CMAKE_COMMAND} -E remove python_file
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/python_binding/exemples $<TARGET_FILE_DIR:quantit> 
                    COMMAND ${CMAKE_COMMAND} -E touch python_file
                    DEPENDS ${PYTHON_EXEMPLES})

endif(pybind11_FOUND)