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
    i for i in range(len(current.parents)) if current.parents[i].name == "mtp4py"
)
repo_path = current.parents[repo_level]
json_file_path = repo_path / "libsetter.json"

with open(json_file_path) as json_file:
    data = json.load(json_file)

# Set the compiler environment variables
CC = os.environ.get("CC", None)
CXX = os.environ.get("CXX", None)
if CC is None:
    os.environ["CC"] = data["compiler"]["CC"]
if CXX is None:
    os.environ["CXX"] = data["compiler"]["CXX"]

MLIP_LIB = os.environ.get("MLIP_LIB", None)
MLIP_DIR = os.environ.get("MLIP_DIR", None)

src_path = repo_path / "src"
with open(src_path / "libsetter.py", "w") as f:
    f.write("# Auto-generated from libsetter.json\n")
    f.write("data = ")
    f.write(json.dumps(data, indent=4))  # Write the JSON data as Python dictionary
    f.write("\n")


if MLIP_DIR is None:
    path_mlip_dir = Path(data["path_to_mlip2"]["MLIP_DIR"])
    if path_mlip_dir.exists():
        MLIP_DIR = str(path_mlip_dir)
    else:
        sys.exit()

if MLIP_LIB is None:
    path_mlip_LIB = Path(data["path_to_mlip2"]["MLIP_LIB"])
    if path_mlip_LIB.exists():
        MLIP_LIB = str(path_mlip_LIB)
    else:
        sys.exit()

print(f"MLIP_LIB = {MLIP_LIB} \nMLIP_DIR={MLIP_DIR}\n\n")

mlip_include_dir = [
    f"{MLIP_DIR}/src/common",
    f"{MLIP_DIR}/src",
    f"{MLIP_DIR}/dev_src",
]

ext_modules = [
    Extension(
        "mtp_cpp2py.core._mtp",
        sources=["mtp_cpp2py/core/_mtp.cpp"],
        include_dirs=[np.get_include(), *mlip_include_dir],
        extra_objects=[MLIP_LIB],
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"],
        language="c++",
    )
]

version = "1.0"
setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    packages=find_packages(exclude=["docs", "tests"]),
)
