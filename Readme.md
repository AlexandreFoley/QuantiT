# QuantiT

## Installation on a Compute Canada Supercomputer

	add python 3.7+.
	add dependencies for cuda (module spider cuda/11.4)
	add a pytorch compatible version of cuda (11.4 seems ok even though it's not explicitly present in pytorch's list, we have to use 11.4 for cudnn on beluga).
	add a version of cudnn compatible with your version of cuda (cudnn 8.2.0 for cuda 11.4 at the moment of writing this)
	add fmt 7
	save your module list (module save [profile_name])
	install pytorch in your userspace (pip install --no-index torch)

Note: if you use vscode to connect to the supercomputer, you might have to erase your vscode setting (~/.vscode-server) anytime you change your default module list in order for it to be loaded when you connect through vscode. VScode seem to reload whatever the default module list was at the first connetion.
## Installation

QuantiT builds on pytorch's tensors, so we must install pytorch first.
To garantee correct compilation in debug mode, we must compile pytorch ourselves. This can take a significant amount of time.
If compiling in release mode, we could make use of precompiled pytorch for our platform. The project isn't set up for that, you're on your own if you want to do that.

This project depends on the {fmt} v7 library, install it with a package manager. (package often called libfmt)

### PyTorch's dependencies
Python3 and some modules are dependencies of PyTorch.
Cuda is an optionnal dependency of PyTorch.

For python, I suggest using anaconda or miniconda, because this make installing some of the additionnal (and optionnal) cuda libraries significantly easier
If you want cuda, install cuda10.2, the latest supported version.

Pytorch can make use of a BLAS and of LAPACK, and require CMake to build (as does QuantiT).
With conda, you can install all the necessary components with the following commands:
	
	# install MKL (intel's blas and lapack implementation) and other things.
	conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi

Add LAPACK support for the GPU if needed

	conda install -c pytorch magma-cuda102

Add cuDNN if you want, version 7.6.5, for cuda10.2, make sure this is right for you.

	conda install cudnn=7.6.5=cuda10.2_0

if you do not wish to (or can't) use conda, you can install the python packages with pip (numpy pyyaml setuptools cffi), lapack and a BLAS with you package manager, and cuda (magma and  cudnn) have to be installed manually.

### build PyTorch
Pytorch is a rather large project, and as such can take a long time to build.
if you do not need the debug symbols, you can [download](https://pytorch.org/get-started/locally/) libtorch in release mode and drop it in \<path to QuantiT\>/extern/pytorch/torch.
If you do so, you must make sure QuantiT is configure accordingly to pytorch (only release currently avaible for most platform). 
Note that pytorch precompiled in debug mode is also available for windows.

Otherwise follow the instruction below.

on the command line, begin with the following:

	cd <path to QuantiT>
	git submodule update --init --recursive

This will download the submodules necessary for QuantiT to compile and work: pytorch and all its' dependencies
We must then compile pytorch.

	cd extern/pytorch

if you want debug symbols:

	export DEBUG=True
	
If you compile the CUDA components, I strongly suggest specifying the architecture of your GPU. 
Otherwise, pytorch will build for a large array of different architecture, this can cause problem when comes time to link the translation units.
	
	export TORCH_GPU_ARCH_LIST="Turing"

The 20XX RTX cards have the "7.5" Turing architecture. Consult [this table](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) to determine the architecture of your GPU.

It can be necessarry to compile Torch in debug in order to debug QuantiT, libtorch tend to be binary incompatible with programs compiled differently.
This may vary by compiler.

if you want only libtorch (torch's python binding will be missing with this):

	export BUILD_PYTHON=False

This is somewhat important if you do not want to interfere with the rest of your python installation. Pytorch's build isn't very clean.

if you're using conda: 

	export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

Now we can build libtorch.

	python3 setup.py build 

### Build QuantiT

Go to QuantiT home directory

	cd <path to QuantiT>

Create a build folder and go into it.

	mkdir build
	cd build

The configure the project.
	
	cmake ..
	
And build with make or ninja, depending on your preference.
You're done.

## Project organization
The project has the following file structure on the top level:

	QuantiT
	├── extern/
	├── include/
	├── Notes/
	├── sources/
	├── tests/
	├──CMakeLists.txt
	├──main.cpp
	└──Readme.md

- extern contain dependencies in the form of git submodules (only pytorch at this moment).
- include contains the headers files: files ending in .h or .hpp
- Notes contain a small latex project documenting facts and thoughts about pytorch relevent to this project.
- sources contains the project's source files: files ending in .cpp
- tests contains the source file for the test executables.
- CMakelists.txt describe the project to cmake for configuration and dependencies
- main.cpp contain the main() function for immediate manual testing of the project
- Readme.md is what you're reading now.

The project is tested using [DocTest](https://github.com/onqtam/doctest), a header only testing framework.
All functions and classes of this project should include tests, and those test shall be built using doctest's framework and located in the headers for that function or class.
Tests are part of the documentation of the project: as well as testing for problems, they are exemple of how to use (or not use, for tests that should fail by design) a component of the project.
