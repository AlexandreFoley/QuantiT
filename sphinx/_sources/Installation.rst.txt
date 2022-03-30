Installation
============

Installation of QuantiT itself should be fairly easy. What you need for it is a compiler and C++ standard library with support for C++17, such as G++-9 or more recent.
Assuming you have torch correctly installed with its dependencies (or you're on OSX), you can 'pip install <link to a release>' or 'pip install git+<link to the git repo>'.
The first option will install a precompiled version of QuantiT that you selected from the release ressources, the second option will compile from the source. If your platform isn't among the precompiled choice, you have to compile from source using the second option.

If you want to use GPU computation you must make sure `Nvidia's CUDA <https://developer.nvidia.com/cuda-downloads>`_ or `AMD ROCm <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_ is installed on your computer and that `torch <https://pytorch.org/get-started/locally/>`_ is correctly installed for the available GPU backend.

On compute canada hardware
--------------------------
The following procedure has been tested on the Beluga Supercomputer.

- add python 3.7+.
- add dependencies for cuda (module spider cuda/11.4)
- add a pytorch compatible version of cuda (11.4 seems ok even though it's not explicitly present in pytorch's list, we have to use 11.4 for cudnn on beluga).
- add a version of cudnn compatible with your version of cuda (cudnn 8.2.0 for cuda 11.4 at the moment of writing this)
- add fmt 7 or 8
- save your module list (module save [profile_name])
- install pytorch in your userspace (pip install --no-index torch)
- install quantit in your userspace (pip install --user git+https://github.com/AlexandreFoley/QuantiT)

Note: if you use vscode to connect to the supercomputer, you might have to erase your vscode setting (~/.vscode-server) anytime you change your default module list in order for it to be loaded when you connect through vscode. VScode seem to reload whatever the default module list was at the first connetion. This is possibly related to .bashrc not being loaded by the vs-code server.