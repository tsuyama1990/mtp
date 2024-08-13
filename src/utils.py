"""Methods to improve the utilities."""

import re

import numpy as np
from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.cell import Cell


def atoms2cfg(atoms, file, append=False):
    """Write the atomic configuration to a .cfg file.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic configuration.
    file : str
        The file path to write the configuration to.
    append : bool, optional
        If True, append to the existing file. If False, create a new file.
    """
    write_f, write_e, write_stress = True, True, True
    mode = "a" if append else "w"
    with open(file, mode) as f:
        try:
            e = atoms.get_potential_energy()
        except Exception:
            write_e = False

        f.write("BEGIN_CFG\n")
        f.write(" Size\n")
        size = len(atoms)
        f.write(f"    {size}\n")
        f.write(" Supercell\n")
        cell = atoms.get_cell()
        for i in range(3):
            f.write(f"  {cell[i][0]:<9}      {cell[i][1]:<9}      {cell[i][2]:<9}\n")

        try:
            fs = atoms.get_forces()
        except Exception:
            write_f = False

        if write_f:
            f.write(
                " AtomData:  id type       cartes_x      cartes_y"
                "      cartes_z           fx          fy          fz\n"
            )
        else:
            f.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z\n")

        pos = atoms.positions
        symbols = atoms.symbols
        for i in range(size):
            aid = i + 1
            atype = Atom(symbols[i]).number
            x, y, z = pos[i]
            if write_f:
                f_x, f_y, f_z = fs[i]
                f.write(
                    f"{aid:>14}{atype:>5}{x:>16.8f}{y:>16.8f}{z:>16.8f}"
                    f"{f_x:>12.6f}{f_y:>12.6f}{f_z:>12.6f}\n"
                )
            else:
                f.write(f"{aid:>14}{atype:>5}{x:>16.8f}{y:>16.8f}{z:>16.8f}\n")

        if write_e:
            f.write(" Energy\n")
            f.write(f"{e:16.6f}\n")

        try:
            stress = atoms.get_stress() * -atoms.get_volume()
        except Exception:
            write_stress = False

        if write_stress:
            f.write(
                " PlusStress:  xx        yy        zz        yz        xz        xy\n"
            )
            f.write(
                f"{stress[0]:.5f}    {stress[1]:.5f}    {stress[2]:.5f}    "
                f"{stress[3]:.5f}    {stress[4]:.5f}    {stress[5]:.5f}\n"
            )

        f.write("END_CFG\n")
        f.write("\n")


def cfg2atoms(file, symbols=None):
    """Read and parse the CFG file to extract energy, forces, and stress.

    Parameters
    ----------
    file : str
        Path to the CFG file.
    symbols : list of str
        List of unique elements present in the system.

    Yield:
    -------
    atoms : ase.Atoms
        The ASE Atoms object with the parsed data.
    """
    with open(file) as f:
        lines = f.read()

    block_pattern = re.compile("BEGIN_CFG\n(.*?)\nEND_CFG", re.S)
    lattice_pattern = re.compile(" SuperCell\n(.*?)\n AtomData", re.S | re.I)
    position_pattern = re.compile("fz\n(.*?)\n Energy", re.S)
    energy_pattern = re.compile(" Energy\n(.*?)\n (?=PlusStress|Stress)", re.S)
    stress_pattern = re.compile("xy\n(.*?)(?=\n|$)", re.S)

    def formatify(string):
        """Convert a space-separated string of numbers to a list of floats."""
        return [float(s) for s in string.split()]

    for block in block_pattern.findall(lines):
        lattice_str = lattice_pattern.findall(block)[0]
        lattice = np.array(list(map(formatify, lattice_str.split("\n"))))
        cell = Cell(lattice)

        position_str = position_pattern.findall(block)[0]
        position = np.array(list(map(formatify, position_str.split("\n"))))
        if symbols is not None:
            chemical_symbols = [symbols[int(ind)] for ind in position[:, 1].astype(int)]
        else:
            chemical_symbols = Atoms(position[:, 1].astype(int)).numbers

        forces = position[:, 5:8]
        positions = position[:, 2:5]

        energy_str = energy_pattern.findall(block)[0]
        energy = float(energy_str.strip())

        stress_str = stress_pattern.findall(block)[0]
        virial_stress = -np.array(list(map(formatify, stress_str.split()))).reshape(
            6,
        )

        atoms = Atoms(
            symbols=chemical_symbols, positions=positions, cell=cell, pbc=True
        )
        volume = atoms.get_volume()
        virial_stress /= volume

        calc = SinglePointCalculator(
            atoms, energy=energy, forces=forces, stress=virial_stress
        )
        atoms.set_calculator(calc)

        yield atoms
