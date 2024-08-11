"""Setup the configurations of cython for C++ programs in pympt.

This setting cannot be done using pyproject.toml.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from Cython.Distutils import build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension

current = Path(__file__).resolve()
repo_level = next(
    i for i in range(len(current.parents)) if current.parents[i].name == "mtp"
)
repo_path = current.parents[repo_level]
json_file_path = repo_path / "libsetter.json"

with open(json_file_path) as json_file:
    data = json.load(json_file)

# Extract compiler settings
cc_compiler = data["compiler"]["CC"]
cxx_compiler = data["compiler"]["CXX"]

# Set the compiler environment variables
os.environ["CC"] = cc_compiler
os.environ["CXX"] = cxx_compiler

environ = os.environ
MLIP_LIB = environ.get("MLIP_LIB", None)
MLIP_DIR = environ.get("MLIP_DIR", None)


if MLIP_DIR is None or MLIP_LIB is None:
    print("can't find MLIP_DIR and MLIP_LIB in environment of system.")
    sys.exit()
else:
    print(f"MLIP_LIB = {MLIP_LIB} \nMLIP_DIR={MLIP_DIR}\n\n")

mlip_include_dir = [
    f"{MLIP_DIR}/src/common",
    f"{MLIP_DIR}/src",
    f"{MLIP_DIR}/dev_src",
]
current = Path(__file__).resolve().parent
ext_modules = [
    Extension(
        "pymtp.core._mtp",
        sources=[f"{current}/core/_mtp.cpp"],
        include_dirs=[np.get_include(), *mlip_include_dir],
        extra_objects=[MLIP_LIB],
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"],
        language="c++",
    )
]

here = os.path.abspath(os.path.dirname(__file__))

version = "1.0"
setup(
    version=version,
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    packages=find_packages(exclude=["docs", "tests"]),
    install_requires=["numpy"],
    extras_require={"all": ["ase"]},
)
