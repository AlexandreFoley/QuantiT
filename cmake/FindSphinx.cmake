
find_program(SPHINX_EXECUTABLE
				NAMES sphinx-build
				DOC "Path to spinx-build")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Sphinx "Failed to find sphinx-build" SPHINX_EXECUTABLE)