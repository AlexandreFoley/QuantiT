# QuanTT

## Installation

If you want to build QuanTT with debug symbols, we must first build pytorch in debug also:
### PyTorch's dependencies
Python3 and some modules are dependencies of PyTorch
Cuda is an optionnal dependency of PyTorch.

For python, I suggest using anaconda or miniconda, because this make installing some of the additionnal (and optionnal) cuda libraries significantly easier
If you want cuda, install cuda10.2, the latest suppoorted version.

Pytorch can make use of a BLAS and of LAPACK, and require CMake to build (as does QuanTT).
With conda, you can install all the necessary components with the following commands:
	
	# install MKL (intel's blas and lapack implementation) and other things.
	conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
Add LAPACK support for the GPU if needed

	conda install -c pytorch magma-cuda102
Add cuDNN if you want, version 7.6.5, for cuda10.2, make sure this is right for you.

	conda install cudnn=7.6.5=cuda10.2_0


### build PyTorch
Pytorch is a rather large project, and as such can take a long time to build.
if you do not need the debug symbols, you can [download](https://pytorch.org/get-started/locally/) libtorch in release mode and drop it in extern/pytorch/torch.
If you do so, you must make sure QuanTT is configure for release. 
Note that precompiled in debug mode is also available for windows.

Otherwise follow the instruction below.

on the command line, begin with the following:

	cd <path to QuanTT>
	git submodule update --init --recursive

This will download the submodules necessary for QuanTT to compile and work: pytorch and all its' dependencies
We must then compile pytorch.

	cd extern/pytorch

if you want debug symbols:

	export DEBUG=True
	
If you compile the CUDA components, I strongly suggest specifying the architecture of your GPU. 
Otherwise, pytorch will build for a large array of different architecture, this can cause problem when comes time to link the translation units.
	
	export TORCH_GPU_ARCH_LIST="Turing"

The latest RTX cards have the "7.5" Turing architecture. Consult [this table](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) to determine the architecture of your GPU.

It can be necessarry to compile Torch in debug in order to debug QuanTT, libtorch tend to be binary incompatible with programs compiled differently.
This may vary by compiler.

if you want only libtorch (torch's python binding will be missing with this):

	export BUILD_PYTHON=False

This is somewhat important if you do not want to interfere with the rest of your python installation. Pytorch's build isn't very clean.

Now we can build libtorch.

	python3 setup.py build 