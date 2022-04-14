


import pathlib as pl

try:
	from .quantit import *
except ImportError as e:
	# This might require elevated permission if the site-package is write protected. basically, if elevated permission are required for pip install, elevated permission are required on first run.
	# This is the post-install script, quantit.so and libQuantit.so aren't expected to be able to find torch's dynamic library 
	# in a portable way by just copying the quantit folder around.
	# This find the torch installation directory and symlink it into quantit install directory whenever quantit.so fails to load torch.
	# if not "cannot open shared object file: No such file or directory" in e.msg:
	# 	#if the error message doesn't relate to missing shared object there's no point in doing the rest, so we raise the execption to the importer.
	# 	raise e
	
	
	init_path = pl.Path(__file__).resolve()
	quantit_path = init_path.parent
	torch_symlink = quantit_path / "lib" /"torch"

	import torch

	cmake_prefix = pl.Path(torch.utils.cmake_prefix_path).resolve() #<torch_dir>/share/cmake/
	torch_lib_path = cmake_prefix.parent.parent / "lib"

	torch_symlink.symlink_to(torch_lib_path,target_is_directory=True)

	from .quantit import *

def cmake_directory():
	return str(pl.Path(__file__).resolve().parent / "share" / "cmake")