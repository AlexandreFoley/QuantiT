

#Based on Sy Brand tutorial at https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/

find_package(Doxygen)

if(Doxygen_FOUND)
    get_target_property(QUANTIT_PUBLIC_HEADER_DIR QuanTT_lib INTERFACE_INCLUDE_DIRECTORIES)
    MESSAGE("header dir ${QUANTIT_PUBLIC_HEADER_DIR}")
    file(GLOB_RECURSE QUANTIT_PUBLIC_HEADERS ${QUANTIT_PUBLIC_HEADER_DIR}/*.h )
    MESSAGE("${QUANTIT_PUBLIC_HEADERS}")


    set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/include)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
    set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)
    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us

    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                       DEPENDS ${QUANTIT_PUBLIC_HEADERS}
                       COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                       MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                       COMMENT "Generating Doxygen documentatino")

    add_custom_target(Doxygen DEPENDS ${DOXYGEN_INDEX_FILE} DEPENDS ${QUANTIT_PUBLIC_HEADERS})


    find_package(Sphinx)

    execute_process(COMMAND python3 -c "import breathe" RESULT_VARIABLE breathe_RETURN OUTPUT_QUIET ERROR_QUIET)
    execute_process(COMMAND python3 -c "import sphinx_rtd_theme" RESULT_VARIABLE sphinx_rtd_theme_RETURN OUTPUT_QUIET ERROR_QUIET)
    if(Sphinx_FOUND AND breathe_RETURN EQUAL 0 AND sphinx_rtd_theme_RETURN EQUAL 0) #Damn cmake is shit.
        set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
        set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
        set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

        # Only regenerate Sphinx when:
        # - Doxygen has rerun
        # - Our doc files have been updated
        # - The Sphinx config has been updated
        add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                           COMMAND ${SPHINX_EXECUTABLE} -b html
                          # Tell Breathe where to find the Doxygen output
                           -Dbreathe_projects.QuantiT=${DOXYGEN_OUTPUT_DIR}/xml
                           ${SPHINX_SOURCE} ${SPHINX_BUILD}
                           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                           DEPENDS
                           # Other docs files you want to track should go here (or in some variable)
                           ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                           ${DOXYGEN_INDEX_FILE}
                           MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                           COMMENT "Generating documentation with Sphinx")


        add_custom_target(Sphinx DEPENDS ${SPHINX_INDEX_FILE})
        add_custom_target(Documentation DEPENDS ${SPHINX_INDEX_FILE} COMMENT "build all the documentation you can")
    else()
        MESSAGE(WARNING "Sphinx + breathe + sphinx_rtd_theme not found, only plain Doxygen documentation available")
        add_custom_target(Documentation DEPENDS ${DOXYGEN_INDEX_FILE} COMMENT "build all the documentation you can")
    endif(Sphinx_FOUND AND breathe_RETURN EQUAL 0 AND sphinx_rtd_theme_RETURN EQUAL 0)
    else()
endif(Doxygen_FOUND)