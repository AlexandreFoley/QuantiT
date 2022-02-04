
import sys
import site 
import os
import pathlib as pl

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages


# trying to figure out the correct rpath to torch from within pip call...
# With a pre-built wheel this will have to be done with system call to overwrite to rpath in the prebuilt libraries.
# The way it's gonna be done: default auto mod uses the first installation it find by searching in the user-site then in the environement site.
# User site need to be skipped if we're in a virtualenv.
# One way to detect a virtualenv we can compare the environment variable VIRTUAL_ENV with the site_package.
# if a virtual env is set and we're runninf from it, then at least one of the lines of site.getsitepackages() will contain the string in the env var.

def get_torch_rpath():
    """find the value for the rpath so that quantit can find torch, this 
    strategy should be good whether we're building or copying a pre-built 
    library. Usage of the resulting string is much more complicated in the
    later case. Note: The easiest portable solution is probably to set a 
    rpath to $ORIGIN/torchlib, and create symlink in the quantit directory ($ORIGIN)"""
    if "--torch-rpath" in sys.argv:
        i = sys.argv.index("--torch-rpath")
        path = sys.argv[i+1]
        plp = pl.path(path)
        if plp.exists():
            return path
        else:
            raise RuntimeError("User supplied path to torch library does not exist")
    paths = []
    if sys.prefix == sys.base_prefix:
        #not a virtual env
        #Those two value are unnaffected by pip's isolation system.
        paths.append(site.getusersitepackages)
    for path in site.getsitepackages():
         paths.append(path)
    for path in paths:
        plp = pl.Path(path)
        q = plp / 'torch' / 'lib'
        if q.exists():
            return os.fspath(q)
    raise RuntimeError("Install script failed to discorver pytorch installation folder. Is pytorch installed?")

torch_rpath = get_torch_rpath()

setup(
    name="quantit",
    version="0.1.0",
    description="QuantiT python bindings",
    author="Alexandre Foley",
    license="All rights reserved",
    packages=find_packages(where="python_binding"),
    package_dir={"": "python_binding"},
    cmake_install_dir="python_binding/quantit",
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    cmake_args=['-DEMIT_PROFILER:BOOL=OFF','-DUSE_EXTERN_TORCH:BOOL=OFF',
    "-DTORCH_INSTALL_RPATH:STRING={}".format(torch_rpath),"-DDISABLE_DOCTEST:BOOL=TRUE"],
    python_requires=">=3.6",
    install_requires = ["torch","numpy",]
)
