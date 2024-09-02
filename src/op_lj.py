"""Optimize LJ potentials."""

from logging import INFO, getLogger

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import sort
from scipy.optimize import fmin
from sklearn import linear_model

from src.ase_calculator_mtp import LammpsLJBuilder

logger = getLogger(__name__)
logger.setLevel(INFO)


class LJTrainer:
    """Trainer class for optimizing Lennard-Jones (LJ) parameters.

    parameters (epsilon and sigma) are optimized using labelled atomic configurations \
        (either energies or forces).

    Parameters
    ----------
    init_epsilon_dict : dict
        Initial dictionary of epsilon values for each element.
    init_sigma_dict : dict
        Initial dictionary of sigma values for each element.
    """

    def __init__(self, init_eps_dict, init_sigma_dict):
        """Initializer of LJ Trainer.

        Parameters:
        ----------
        init_eps_dict : dict
            Initial dictionary of epsilon values for each element.
        init_sigma_dict : dict
            Initial dictionary of sigma values for each element.
        """
        self.init_eps_dict = init_eps_dict
        self.init_sigma_dict = init_sigma_dict
        lj = LammpsLJBuilder(init_eps_dict, init_sigma_dict)
        self.element_list_sorted, self.ele_num = lj.get_sorted_ele(init_eps_dict.keys())

    def run_minimize(self, minimized_by, labelled_list_atoms, maxiter=100):
        """Runs optimization to minimize either energy or forces.

        Parameters:
        ----------
        minimized_by : str
            Defines whether the objective is to minimize 'energies' or 'forces'.
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects with known energies or forces.
        """
        opt_pars = [self.init_eps_dict[ele] for ele in self.element_list_sorted]
        opt_pars += [self.init_sigma_dict[ele] for ele in self.element_list_sorted]

        if minimized_by == "energies":
            self.build_atom_offset_ds(labelled_list_atoms)
            objective_func = self.objective_by_energy
            init_dict_atom_eng = self.chemipot
            augment = (
                self.element_list_sorted,
                labelled_list_atoms,
                init_dict_atom_eng,
            )

        elif minimized_by == "forces":
            objective_func = self.objective_by_forces
            augment = (
                self.element_list_sorted,
                labelled_list_atoms,
            )

        parameters_optimized = fmin(
            objective_func, opt_pars, args=augment, maxiter=maxiter
        )
        logger.info("optimization done")

        eps = {}
        sigma = {}
        for i, ele in enumerate(self.element_list_sorted):
            eps[ele] = parameters_optimized[i]
            sigma[ele] = parameters_optimized[i + len(self.element_list_sorted)]

        return eps, sigma

    @staticmethod
    def objective_by_energy(
        opt_paras, unique_atoms, labelled_list_atoms, dict_atom_eng
    ):
        """Objective function for LJ.

        to minimizing energy differences between labelled \
            data and LJ-predicted data.

        Parameters:
        ----------
        opt_paras : list of float
            Optimized epsilon and sigma values.
        unique_atoms : list of str
            List of unique atom symbols.
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects with known energies.
        dict_atom_eng : dict
            Dictionary of atomic chemical potentials.

        Returns:
        -------
        float
            Root mean square error (RMSE) between labelled and LJ energies.
        """
        labelled_eng = []
        lj_eng = []

        # build dicts for lammps calculators
        eps = {}
        sigma = {}
        for i, ele in enumerate(unique_atoms):
            eps[ele] = opt_paras[i]
            sigma[ele] = opt_paras[i + len(unique_atoms)]

        # lammps calculator
        lj = LammpsLJBuilder(dict_eps=eps, dict_sigma=sigma)
        lj_calc = lj.get_calculator()
        for atoms in labelled_list_atoms:
            labelled_eng.append(atoms.get_potential_energy())
            atoms.calc = lj_calc
            eng = atoms.get_potential_energy()
            offset = np.array(
                [
                    dict_atom_eng[ele] * np.sum(atoms.symbols == ele)
                    for ele in dict_atom_eng.keys()
                ]
            ).sum()
            lj_eng.append(eng + offset)
        diff = np.array(labelled_eng) - np.array(lj_eng)
        rsme = np.linalg.norm(diff)

        logger.info(f"objective : {rsme}")
        logger.info(f"epsilon : {eps}")
        logger.info(f"sigma : {sigma}")

        return rsme

    @staticmethod
    def objective_by_forces(opt_paras, unique_atoms, labelled_list_atoms):
        """Objective function for LJ.

        to minimizing forces differences between labelled \
            data and LJ-predicted data.

        Parameters:
        ----------
        opt_paras : list of float
            Optimized epsilon and sigma values.
        unique_atoms : list of str
            List of unique atom symbols.
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects with known forces.

        Returns:
        -------
        float
            Root mean square error (RMSE) between labelled and LJ forces.
        """
        diff_forces = []

        # build dicts for lammps calculators
        eps = {}
        sigma = {}
        for i, ele in enumerate(unique_atoms):
            eps[ele] = opt_paras[i]
            sigma[ele] = opt_paras[i + len(unique_atoms)]

        # lammps calculator
        lj = LammpsLJBuilder(dict_eps=eps, dict_sigma=sigma)
        lj_calc = lj.get_calculator()

        for atoms in labelled_list_atoms:
            dft_forces = atoms.get_forces()
            atoms.calc = lj_calc
            lj_forces = atoms.get_forces()
            diff_force = dft_forces - lj_forces
            diff_forces += diff_force.to_list()
        diff_forces = np.array(diff_forces)
        rsme = np.linalg.norm(diff_forces)

        logger.info(f"objective : {rsme}")
        logger.info(f"epsilon : {eps}")
        logger.info(f"sigma : {sigma}")

        return rsme

    def atom_ds_builder(self, labelled_list_atoms):
        """Generator to build a dataset from labelled atomic configurations.

        Parameters:
        ----------
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects.

        Yields:
        ------
        list
            A list containing index, sorted atomic symbols, potential energy,
            and Atoms object.
        """
        for i, atoms in enumerate(labelled_list_atoms):
            yield [i, sort(atoms).symbols, atoms.get_potential_energy(), atoms]

    def get_lowest_eng_in_compositions(self, labelled_list_atoms):
        """Returns the lowest energy configuration for each atomic composition.

        Parameters:
        ----------
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects.

        Returns:
        -------
        pandas.DataFrame
            DataFrame with the lowest energy configuration for each composition.
        """
        df = pd.DataFrame(
            list(self.atom_ds_builder(labelled_list_atoms)),
            columns=["compositions", "eng_eV", "Atoms.Obj"],
            index=0,
        )

        def get_mins(df_dummy):
            min_id = df_dummy["eng_eV"].argmin()
            return df.loc[min_id]

        df_min = df.groupby(by="compositions", as_index=False).apply(get_mins)
        return df_min

    def get_atom_num(self, symbols):
        """Yields the number of atoms for each element in a given atomic configuration.

        Parameters:
        ----------
        symbols : list of str
            List of atomic symbols.

        Yields:
        ------
        int
            The count of atoms for each element.
        """
        atoms = Atoms(symbols)
        for ele in self.element_list_sorted:
            yield (atoms.symbols == ele).sum()

    def build_atom_offset_ds(self, labelled_list_atoms):
        """Builds a dataset of atomic offsets.

        Offsets are calculated based on chemical potentials \
            acquired by the lowest energy configurations.

        Parameters:
        ----------
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects.
        """
        df_min = self.get_lowest_eng_in_compositions(labelled_list_atoms)
        train_y = df_min["eng_eV"].values
        train_x = []
        for symbols in df_min["compositions"]:
            train_x.append(list(self.get_atom_num(symbols)))

        train_x = np.array(train_x)
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(train_y, train_x)
        self.chemipot = {
            self.element_list_sorted[i]: regr.coef_[i]
            for i in range(len(self.element_list_sorted))
        }
        logger.info("estimated rough chemical potentials")
        logger.info(f"{self.chemipot}")
