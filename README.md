# mtp4py

[Memont Tensor Potentials (MLIP-2)](https://gitlab.com/ashapeev/mlip-2.git)
is wrapped for python.
The code developd by [pymtp](https://github.com/hlyang1992/pymtp.git) is further wraped
to improve the utility by python.
Appropriate citations for above repos are required to use.

## Training

```python
from pymtp.core import MTPCalactor
from pymtp.core import PyConfiguration
from ase.io import read

calc = CalculatorMtp4Py("pot")
atoms = read("POSCAR")

atoms.calc = calc

print(atoms.get_potential_energy())
print(atoms.get_forces())
print(atoms.get_stress())
```

## Inferences

```python
from pymtp.core import MTPCalactor
from pymtp.core import PyConfiguration
from ase.io import read

calc = CalculatorMtp4Py("path_to_pot_file")
atoms = read("path_to_POSCAR")

atoms.calc = calc

print(atoms.get_potential_energy())
print(atoms.get_forces())
print(atoms.get_stress())
```

## Features

This usage of ASE calculator is essential to run MD using ASE.
Not essential but training / dataset building can be also managed by python environment.

## Requirement

[Memont Tensor Potentials](https://gitlab.com/ashapeev/mlip-2.git).
This repo is not compatible with [mlip-3](https://gitlab.com/ashapeev/mlip-3.git),
so be careful.

## Installation

First, clone this repo.

```bash
git clone https://github.com/tsuyama1990/mtp.git
```

### MLIP-2

If you do not have [MLIP-2](https://gitlab.com/ashapeev/mlip-2.git)
clone it in any path.
For the installation pls follow
[README in MLIP-2](https://gitlab.com/ashapeev/mlip-2.git).

* g++, gcc, gfortran, mpicxx
* Alterlatively: the same set for Intel compilers (tested with the 2017 version)
* make
are necessary to compile MLIP.

For executables, you need to make below.

* mlp (make mlp)
* libinterface (make libinterface)

are essential at leaset.

### Set configuration

Edit [json files to set paths and compilers](libsetter.json).

* MLIP_DIR : path to the mlip2 directory
* MLIP_LIB : "path to the lib_mlip_interface.a file
* CC : C compiler, e.g. 'gcc-11', 'icpc'
* CXX" : C++ compiler e.g. 'g++-11', 'icpc'

### Setup

After this, go ahead installtion.
2 stesps

```bash
cd <go to this repo>
source startup_local.sh
```

***USE SOURCE not BASH***

```bash
cd <go to this repo> / pytmp
poetry run python setup.py install
```

This process is necessary because pyproject.toml cannot handle cython.

## Author

* Tomoyuki Tsuyama
* <tomoyuki.tsuyaman@gmail.com>

## License

"mtp4py" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
