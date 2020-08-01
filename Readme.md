# QuanTT

## Installation

If you want to build QuanTT with debug symbols, we must first build pytorch in debug also:

### build pytorch
Pytorch is a rather large project, and as such can take a long time to build.
if you do not need the debug symbols, you can [download](https://pytorch.org/get-started/locally/) libtorch in release mode and drop it in extern/pytorch/torch.
If you do so, you must make sure QuanTT is configure for release. 
Note that precompiled in debug mode is also available for windows.

Otherwise follow the instruction below.

on the command line, begin with the following:

	cd <path to QuanTT>
	git submodule update --recursive

This will download the submodules necessary for QuanTT to compile and work: pytorch and all its' dependencies
We must then compile pytorch.


	cd extern/pytorch

if you want debug symbols:

	export CMAKE_BUILD_TYPE=Debug

It can be necessarry to compile Torch in debug in order to debug QuanTT, libtorch tend to be binary incompatible with programs compiled differently.
This can vary by compiler.

if you want only libtorch (torch's python binding will be missing with this):

	export BUILD_PYTHON=False

This is somewhat important if you do not want to interfere with the rest of your python installation. Pytorch's build isn't very clean.

Now we can build libtorch.

	python3 setup.py build 