# QuantiT

## Installation on a Compute Canada Supercomputer

	add python 3.7+.
	add dependencies for cuda (module spider cuda/11.4)
	add a pytorch compatible version of cuda (11.4 seems ok even though it's not explicitly present in pytorch's list, we have to use 11.4 for cudnn on beluga).
	add a version of cudnn compatible with your version of cuda (cudnn 8.2.0 for cuda 11.4 at the moment of writing this)
	add fmt 7
	save your module list (module save [profile_name])
	install pytorch in your userspace (pip install --no-index torch)
	install quantit in your userspace (pip install --user git+https://github.com/AlexandreFoley/QuantiT)

Note: if you use vscode to connect to the supercomputer, you might have to erase your vscode setting (~/.vscode-server) anytime you change your default module list in order for it to be loaded when you connect through vscode. VScode seem to reload whatever the default module list was at the first connetion.
	This is possibly related to .bashrc not being loaded by the vs-code server.
## Installation

QuantiT builds on pytorch's tensors, so we must install pytorch first.
To garantee correct compilation in debug mode, we must compile pytorch ourselves. This can take a significant amount of time.
If compiling in release mode, we could make use of precompiled pytorch for our platform. The project isn't set up for that, you're on your own if you want to do that.

This project depends on the {fmt} v7 library, install it with a package manager. (package often called libfmt)
You can skip the installation of {fmt}, if you do, the project will pull it from github at bundle it inside quantit's build.

### PyTorch's dependencies
Python3 and some modules are dependencies of PyTorch.
Cuda is an optionnal dependency of PyTorch.

For python, I suggest using anaconda or miniconda, because this make installing some of the additionnal (and optionnal) cuda libraries significantly easier.
But you don't have to.

Pytorch can make use of a BLAS and of LAPACK, and require CMake to build (as does QuantiT).
With conda, you can install all the necessary components with the following commands:
	
	# install MKL (intel's blas and lapack implementation) and other things.
	conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi

Add LAPACK support for the GPU if needed

	conda install -c pytorch magma-cuda102

Add cuDNN if you want, version 7.6.5, for cuda10.2, make sure this is right for you.

	conda install cudnn=7.6.5=cuda10.2_0

If you do not wish to (or can't) use conda, you can install the python packages with pip (wheel numpy pyyaml setuptools cffi), lapack and a BLAS with you package manager, and cuda (magma and  cudnn) have to be installed manually. See these installation instructions for [magma](https://icl.cs.utk.edu/projectsfiles/magma/doxygen/installing.html) and [cudnn](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

### build PyTorch

Go to pytorch's [getting started page](https://pytorch.org/get-started/locally/) and select your system, package, compute platform and the python language. Then execute the supplied package manager command.

## Project organization
The project has the following file structure on the top level:

	QuantiT
	├── Design documents/
	├── Documentation/
	├── extern/
	├── include/
	├── Notes/
	├── python_binding/
	├── sources/
	├── tests/
	├──CMakeLists.txt
	├──main.cpp
	└──Readme.md

- Design Document contain a small latex document that aims to explain the rationnal behind's btensor structure.
- Documentation contain the necessary parts to build the doxygen doc from cmake, and integrate that in a (nicer looking) spinx generated documentation.
- extern contain dependencies in the form of git submodules (only pytorch at this moment).
- include contains the headers files: files ending in .h or .hpp
- Notes contain a small latex project documenting facts and thoughts about pytorch relevent to this project.
- python_binding contains the cpp code specific to the python interface
- sources contains the project's source files: files ending in .cpp
- tests contains the source file for the test executables.
- CMakelists.txt describe the project to cmake for configuration and dependencies
- main.cpp contain the main() function for immediate manual testing of the project
- Readme.md is what you're reading now.

The project is tested using [DocTest](https://github.com/onqtam/doctest), a header only testing framework.
All functions and classes of this project should include tests, and those test shall be built using doctest's framework and located in the headers for that function or class.
Tests are part of the documentation of the project: as well as testing for problems, they are exemple of how to use (or not use, for tests that should fail by design) a component of the project.
