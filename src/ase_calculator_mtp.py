"""Define Calculator for MTP."""

from itertools import combinations_with_replacement

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lammpslib import LAMMPSlib

from mtp_cpp2py.core import MTPCalactor, PyConfiguration

ALL_CHANGES = tuple(all_changes)


class AseCalculatorMtp(Calculator):
    """ASE Calculator interface for the Moment Tensor Potential (MTP).

    This class provides methods to calculate energy, forces, and stress using MTP.
    """

    def __init__(self, mtp_path, **kwargs):
        """Initialize the MTP calculator.

        Parameters
        ----------
        mtp_path : pathlike
            Path to the MTP potential file.
        **kwargs
            Additional keyword arguments for the ASE Calculator.
        """
        self.pymtp_calc = MTPCalactor(str(mtp_path))
        super().__init__(**kwargs)
        self.implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces", "stress"],
        system_changes=ALL_CHANGES,
    ):
        """Perform the calculation of the specified properties.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            ASE atoms object for which the calculation is performed.
        properties : list of str, optional
            List of properties to calculate. Default is ["energy", "forces", "stress"].
        system_changes : list of str, optional
            List of changes in the system. Default is ALL_CHANGES.
        """
        super().calculate(atoms, properties, system_changes)

        mtp_atoms = PyConfiguration.from_ase_atoms(atoms)
        self.pymtp_calc.calc(mtp_atoms)

        self.results["energy"] = mtp_atoms.energy
        self.results["forces"] = mtp_atoms.force
        self.results["stress"] = mtp_atoms.stresses


class AseCalculatorMtpDLJ(AseCalculatorMtp):
    """ASE Calculator interface for a hybrid MTP & LJ.

    This class extends the AseCalculatorMtp class to include the Lennard-Jones potential
    in addition to MTP. It calculates energy, forces,
    and stress using a combination of both MTP and LJ potentials.
    """

    def __init__(self, mtp_path, dict_eps, dict_sigma, **kwargs):
        """Initialize the hybrid MTP-LJ calculator.

        Parameters
        ----------
        mtp_path : pathlike
            Path to the MTP potential file.
        dict_eps : dict
            Dictionary of epsilon values for each element
            in the Lennard-Jones potential.
        dict_sigma : dict
            Dictionary of sigma values for each element in the Lennard-Jones potential.
        **kwargs
            Additional keyword arguments for the ASE Calculator.
        """
        self.lj_calc = LammpsLJBuilder().get_calculator(
            dict_eps=dict_eps, dict_sigma=dict_sigma
        )
        super().__init__(mtp_path, **kwargs)

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces", "stress"],
        system_changes=ALL_CHANGES,
    ):
        """Perform the calculation using both MTP and LJ potentials.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            ASE Atoms object for which the calculation is performed.
        properties : list of str, optional
            List of properties to calculate. Default is ["energy", "forces", "stress"].
        system_changes : list of str, optional
            List of changes in the system. Default is ALL_CHANGES.
        """
        super().calculate(atoms, properties, system_changes)

        mtp_atoms = PyConfiguration.from_ase_atoms(atoms)
        self.pymtp_calc.calc(mtp_atoms)

        lj_atoms = atoms.copy()
        lj_atoms.calc = self.lj_calc

        self.results["energy"] = mtp_atoms.energy + lj_atoms.calc.get_potential_energy()
        self.results["forces"] = mtp_atoms.force + lj_atoms.calc.get_forces()
        self.results["stress"] = mtp_atoms.stresses + lj_atoms.calc.get_stress()


class LammpsLJBuilder:
    """A class to generate LAMMPS Lennard-Jones potentials.

    The parameters for homogeneous element pairs are determined
    from the dictionary-type argument.
    The parameters for heterogeneous pairs are estimated using
    the Lorentz-Berthelot combining rules.

    """

    def get_calculator(self, dict_eps, dict_sigma):
        """Builds and returns a LAMMPS calculator.

        Parameters:
        ----------
        dict_eps : dict
            Dictionary mapping element symbols to epsilon values (interaction strength).
        dict_sigma : dict
            Dictionary mapping element symbols to sigma values (size of atoms).

        Returns:
        -------
        LAMMPSlib
            A LAMMPS calculator with the defined LJ potential.
        """
        self.dict_eps = dict_eps
        self.dict_sigma = dict_sigma

        lmpcmds, atom_types = self.build_pair_style()
        lammps_calculator = LAMMPSlib(
            lmpcmds=lmpcmds, atom_types=atom_types, keep_alive=True
        )
        return lammps_calculator

    def build_pair_style(self):
        """Constructs LAMMPS pair style commands.

        The commands are being built based on the element types
        and their epsilon and sigma values.

        Returns:
        -------
        list of str
            LAMMPS commands for setting the LJ potential.
        dict
            A dictionary mapping element symbols to numerical atom types.
        """
        element_list_sorted, ele_num = self.get_sorted_ele(self.dict_eps.keys())
        possible_combination_list = self.get_possible_combinations(element_list_sorted)

        # Initialize LAMMPS command. Cutoff is set to 3 * max_sigma.
        max_cutoff = max(self.dict_sigma.values())
        lammps_command = [f"pair_style lj/cut {3 * max_cutoff}"]

        # Adding Pair styles.
        for pair in possible_combination_list:
            # epsilon
            e1 = self.dict_eps.get(pair[0], 1)
            e2 = self.dict_eps.get(pair[1], 1)
            # sigma
            s1 = self.dict_sigma.get(pair[0], 1)
            s2 = self.dict_sigma.get(pair[1], 1)
            # Combine them using Lorentz-Berthelot combining rules.
            epsilon = np.sqrt(e1 * e2)
            sigma = (s1 + s2) / 2
            individual_cutoff = sigma * 3
            # Building pair coeff command.
            pair_coeff = (
                f"pair_coeff {ele_num[pair[0]]} {ele_num[pair[1]]} "
                f"{epsilon} {sigma} {individual_cutoff}"
            )
            lammps_command.append(pair_coeff)

        return lammps_command, ele_num

    def get_sorted_ele(self, ele_list):
        """Sorts the element list based on atomic numbers.

        Parameters
        ----------
        ele_list : iterable
            List of element symbols to sort.

        Returns:
        -------
        list
            Sorted list of element symbols.
        dict
            Dictionary mapping element symbols to their atom types (numerical).
        """
        unique_list = list(set(ele_list))
        sort_index = np.argsort(Atoms(unique_list).numbers)
        unique_list_sorted = list(Atoms(unique_list)[sort_index].symbols)
        ele_num = {ele: idx + 1 for idx, ele in enumerate(unique_list_sorted)}
        return unique_list_sorted, ele_num

    def get_possible_combinations(self, element_list_sorted):
        """Generates all possible combinations (with repetition) of the element types.

        Parameters
        ----------
        element_list_sorted : list of str
            List of sorted element symbols.

        Returns:
        -------
        list of tuple
            List of tuples representing possible element pairs.
        """
        return list(combinations_with_replacement(element_list_sorted, 2))
