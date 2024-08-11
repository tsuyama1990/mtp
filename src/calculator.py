"""Define Calculator for MTP."""

import os
import re

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.cell import Cell


class MtpCalculator(Calculator):
    """ASE Calculator interface for the Moment Tensor Potential (MTP).

    This class provides methods to calculate energy, forces, and stress using MTP.
    """

    def __init__(self, mtp="pot.mtp", **kwargs):
        """Initialize the MTP calculator.

        Parameters
        ----------
        mtp : str, optional
            Path to the MTP potential file. Default is "pot.mtp".
        **kwargs
            Additional keyword arguments for the ASE Calculator.
        """
        self.mtp = mtp
        super().__init__(**kwargs)
        self.implemented_properties = ["energy", "forces", "stress"]

    def initialize(self, atoms):
        """Initialize the atomic numbers for the given atoms object.

        Parameters
        ----------
        atoms : ase.Atoms
            ASE atoms object containing atomic information.
        """
        self.numbers = atoms.get_atomic_numbers()

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """Perform the calculation of the specified properties.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            ASE atoms object for which the calculation is performed.
        properties : list of str, optional
            List of properties to calculate. Default is ["energy"].
        system_changes : list of str, optional
            List of changes in the system. Default is all_changes.
        """
        super().calculate(atoms, properties, system_changes)

        if "numbers" in system_changes:
            self.initialize(self.atoms)

        file = os.path.join(self.parameters.tmp_folder, "in.cfg")
        output = os.path.join(self.parameters.tmp_folder, "out.cfg")
        _atoms = atoms.copy()
        atoms_to_cfg(_atoms, file)

        os.system(f"mlp calc-efs {self.mtp} {file} {output}")
        energy, forces, stress = read_cfg(output)
        self.results["energy"] = energy
        self.results["forces"] = np.array(forces)
        self.results["stress"] = np.array(stress)


def atoms_to_cfg(atoms, file, unique_elements):
    """Convert ASE atoms object to CFG format required by MTP.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE atoms object to be converted.
    file : str
        Path to the output CFG file.
    unique_elements : list of str
        List of unique elements present in the system.
    """
    write_f, write_e = True, True
    with open(file, "w") as f:
        ele_dict = {ele.capitalize(): int(i) for i, ele in enumerate(unique_elements)}
        try:
            e = atoms.get_potential_energy()
        except Exception:
            write_e = False
        f.write("BEGIN_CFG\n")
        f.write(" Size\n")
        size = len(atoms)
        f.write(f"    {int(size)}\n")
        f.write(" Supercell\n")
        cell = atoms.get_cell()
        f.write(
            f"          {cell[0][0]}      {cell[0][1]}      {cell[0][2]}\n"
            f"          {cell[1][0]}      {cell[1][1]}      {cell[1][2]}\n"
            f"          {cell[2][0]}      {cell[2][1]}      {cell[2][2]}\n"
        )
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
            aid = int(i + 1)
            atype = ele_dict[symbols[i]]
            x, y, z = pos[i]
            if write_f:
                f_x, f_y, f_z = fs[i]
                f.write(
                    f"{aid:>14}{atype:>5}{x:>16.8f}{y:>16.8f}{z:>16.8f}{f_x:>12.6f}{f_y:>12.6f}{f_z:>12.6f}\n"
                )
            else:
                f.write(f"{aid:>14}{atype:>5}{x:>16.8f}{y:>16.8f}{z:>16.8f}\n")
        if write_e:
            f.write(" Energy\n")
            f.write(f"{e:16.6f}\n")
        f.write("END_CFG\n\n")


def read_cfg(file, symbols):
    """Read and parse the CFG file to extract energy, forces, and stress.

    Parameters
    ----------
    file : str
        Path to the CFG file.
    symbols : list of str
        List of unique elements present in the system.

    Returns:
    -------
    energy : float
        The energy of the system.
    forces : list of list of float
        The forces on the atoms.
    virial_stress : numpy.ndarray
        The virial stress tensor.
    """
    with open(file) as f:
        lines = f.read()
    block_pattern = re.compile("BEGIN_CFG\n(.*?)\nEND_CFG", re.S)
    lattice_pattern = re.compile("SuperCell\n(.*?)\n AtomData", re.S | re.I)
    position_pattern = re.compile("fz\n(.*?)\n Energy", re.S)
    energy_pattern = re.compile("Energy\n(.*?)\n (?=PlusStress|Stress)", re.S)
    stress_pattern = re.compile("xy\n(.*?)(?=\n|$)", re.S)

    def formatify(string):
        return [float(s) for s in string.split()]

    for block in block_pattern.findall(lines):
        lattice_str = lattice_pattern.findall(block)[0]
        lattice = np.array(list(map(formatify, lattice_str.split("\n"))))
        cell = Cell(lattice)
        volume = cell.volume
        position_str = position_pattern.findall(block)[0]
        position = np.array(list(map(formatify, position_str.split("\n"))))
        # types = position[:, 1].tolist()
        # chemical_symbols = [symbols[int(ind)] for ind in types]
        forces = position[:, 5:8].tolist()
        position = position[:, 2:5]
        energy_str = energy_pattern.findall(block)[0]
        energy = float(energy_str.lstrip())
        stress_str = stress_pattern.findall(block)[0]
        virial_stress = (
            -np.array(list(map(formatify, stress_str.split()))).reshape(
                6,
            )
            / volume
        )
    return energy, forces, virial_stress
