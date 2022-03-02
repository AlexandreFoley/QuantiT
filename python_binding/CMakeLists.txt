



if(pybind11_FOUND)

  MESSAGE("PYTORCH LIB" ${TORCH_INSTALL_PREFIX})
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
  MESSAGE("TORCH LIB: " ${TORCH_PYTHON_LIBRARY}  "<>" ${PYTHON_TORCH_LIBRARY})

  #wraping it in a target library will save us grief if i ever attempt to autogenerate a .cmake for the quantit python module
  add_library(torch_python SHARED IMPORTED)
  set_target_properties(torch_python PROPERTIES
    IMPORTED_LOCATION ${TORCH_PYTHON_LIBRARY}
  )

  add_library(quantit MODULE quantt.cpp conserved.cpp linalg.cpp networks.cpp operators.cpp algorithm.cpp)
  target_link_libraries(quantit PUBLIC pybind11::module torch_python QuantiT)
  set_target_properties(quantit PROPERTIES PREFIX "") #remove the "lib" from the resulting shared object

  set(EX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/exemples)
  set(PYTHON_EXEMPLES EX_DIR/dmrg.py)

  # add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/python_binding/python_file
  #                   COMMAND ${CMAKE_COMMAND} -E remove python_file
  #                  COMMAND ${CMAKE_COMMAND} -E copy_directory
  #                      ${CMAKE_SOURCE_DIR}/python_binding/exemples $<TARGET_FILE_DIR:quantit> 
  #                   COMMAND ${CMAKE_COMMAND} -E touch python_file
  #                   DEPENDS ${PYTHON_EXEMPLES})
  file(GLOB PY_EX "${EX_DIR}/*.py" )
  add_custom_command(TARGET quantit
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${PY_EX}
                   $<TARGET_FILE_DIR:quantit>)
if(APPLE)
set(ORIGIN "@loader_path")
else()
set(ORIGIN "$ORIGIN") ##needs to be tested. LINUX!
endif()
 if(SKBUILD)
   install(TARGETS quantit DESTINATION .)
   set_target_properties(quantit PROPERTIES INSTALL_RPATH "${ORIGIN}/lib/torch;${ORIGIN}/lib")
   #The result of this isn't good... looks like a temporary folder.
   message(STATUS "Using scikit-build, RPATH:${TORCH_INSTALL_RPATH}")
 endif()

endif(pybind11_FOUND)