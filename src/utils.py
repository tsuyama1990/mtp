"""Methods to improve the utilities."""

import re
from pathlib import Path

import numpy as np
from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.cell import Cell
from ase.db import connect

from src import libsetter


def atoms2cfg(atoms, file, bool_delta_learing=False, append=False, lammps_lj_calc=None):
    """Write the atomic configuration to a .cfg file.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic configuration to write to the file.
    file : str
        The file path where the configuration will be written.
    bool_delta_learing : bool, optional
        If True, calculate and write energy and forces
            subtracting Lennard-Jones potential as a first approximation.
        If False, standard energy and forces are calculated. Default is False.
    append : bool, optional
        If True, append to the existing file.
        If False, overwrite the file. Default is False.
    lammps_lj_calc: ase.calculators
        Calculator for Lennard Jones, characterised by parameters.

    Returns:
    -------
    None
        This function does not return any value.

    Raises:
    ------
    Exception
        If there are issues with the potential energy, forces, or stress calculations.

    """
    write_f, write_e, write_stress = True, True, True
    mode = "a" if append else "w"

    if bool_delta_learing:
        lj_atoms = atoms.copy()
        lj_atoms.calc = lammps_lj_calc

    with open(file, mode) as f:
        try:
            if bool_delta_learing:
                e = atoms.get_potential_energy() - lj_atoms.get_potential_energy()
            else:
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
            if bool_delta_learing:
                forces = atoms.get_forces() - lj_atoms.get_forces()

            else:
                forces = atoms.get_forces()

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
                force_x, force_y, force_z = forces[i]
                f.write(
                    f"{aid:>14}{atype:>5}{x:>16.8f}{y:>16.8f}{z:>16.8f}"
                    f"{force_x:>12.6f}{force_y:>12.6f}{force_z:>12.6f}\n"
                )
            else:
                f.write(f"{aid:>14}{atype:>5}{x:>16.8f}{y:>16.8f}{z:>16.8f}\n")

        if write_e:
            f.write(" Energy\n")
            f.write(f"{e:16.6f}\n")

        try:
            if bool_delta_learing:
                stress = (
                    atoms.get_stress() * -atoms.get_volume()
                    - lj_atoms.get_stress() * -lj_atoms.get_volume()
                )
            else:
                stress = atoms.get_stress() * -atoms.get_volume()

        except Exception:
            virial_stress_x_vol = get_virial_x_vol(atoms)

            if bool_delta_learing:
                stress = (
                    virial_stress_x_vol - lj_atoms.get_stress() * -lj_atoms.get_volume()
                )
            else:
                stress = virial_stress_x_vol

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


def get_virial_x_vol(atoms):
    """Get virial stress x volume with voigt style.

    Parameters
    ----------
    atoms : Ase.atoms
        Atoms to calculate the virial.
    """
    forces = atoms.get_forces()
    positions = atoms.get_positions()
    virial_stress_x_vol = np.zeros((3, 3))

    for i in range(len(atoms)):
        r = positions[i]
        f = forces[i]

        virial_stress_x_vol += np.outer(r, f)

    virial_x_vol_voigt = np.array(
        [
            virial_stress_x_vol[0, 0],
            virial_stress_x_vol[1, 1],
            virial_stress_x_vol[2, 2],
            virial_stress_x_vol[1, 2],
            virial_stress_x_vol[0, 2],
            virial_stress_x_vol[0, 1],
        ]
    )
    return virial_x_vol_voigt


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
    energy_pattern = re.compile(" Energy\n(.*?)\n (?=PlusStress|Stress|Feature)", re.S)
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


def get_unique_element_from_cfg(cfg_file_path):
    """Extract unique atomic elements from a CFG file.

    This function reads atomic configurations from a CFG file, extracts
    the chemical symbols for all atoms, and returns a sorted list of unique
    elements present in the dataset.

    Parameters:
    ----------
    cfg_file_path : Path
        Path to the CFG file containing atomic configurations.

    Returns:
    -------
    list
        A sorted list of unique atomic elements found in the CFG file,
        represented by their chemical symbols.
    """
    unique_symbols = []
    for atoms in cfg2atoms(cfg_file_path):
        sym = np.array(atoms.symbols)
        unique_symbols += list(set(sym))

    unique_element = np.array(list(set(unique_symbols)))
    unique_element_sort = np.argsort(Atoms(unique_element).numbers)
    return unique_element_sort.tolist()


def get_unique_element_from_db(db_path):
    """Extract unique atomic elements from a ase_db.

    This function reads atomic configurations from a ase_db, extracts
    the chemical symbols for all atoms, and returns a sorted list of unique
    elements present in the dataset.

    Parameters:
    ----------
    db_path : Path
        Path to the ase_db containing atomic configurations.

    Returns:
    -------
    list
        A sorted list of unique atomic elements found in the ase_db,
        represented by their chemical symbols.
    """
    db = connect(db_path)
    unique_symbols = []
    for row in db.select():
        atoms = row.toatoms()
        unique_symbols += list(set(atoms.symbols))

    unique_symbols = list(set(unique_symbols))
    sort_index = np.argsort(Atoms(unique_symbols).numbers)
    unique_list_sorted = list(Atoms(unique_symbols)[sort_index].symbols)
    ele_num = {ele: idx + 1 for idx, ele in enumerate(unique_list_sorted)}
    return ele_num


def find_mlip_dir():
    """Finds the directory path for MLIP.

    This function identifies the path to the MLIP directory
    by searching the file structure for a 'mtp4py' repository
    and loading the directory path from a JSON file.

    Raises:
    ------
    FileNotFoundError
        If the MLIP directory does not exist.

    Returns:
    -------
    Path
        A `Path` object pointing to the MLIP directory.
    """
    dict_paths = libsetter.data
    path_mlip_dir = Path(dict_paths["path_to_mlip2"]["MLIP_DIR"])

    if not path_mlip_dir.exists():
        raise FileNotFoundError(f"MLIP directory not found: {path_mlip_dir}")

    return path_mlip_dir
