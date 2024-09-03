"""Optimize LJ potentials."""

import logging

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import sort
from scipy.optimize import minimize
from sklearn import linear_model

from src.ase_calculator_mtp import LammpsLJBuilder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LJTrainer:
    """Trainer class for optimizing Lennard-Jones (LJ) parameters.

    mama
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
        self.element_list_sorted, self.ele_num = LammpsLJBuilder().get_sorted_ele(
            init_eps_dict.keys()
        )

    def run_minimize(
        self, labelled_list_atoms, minimized_by="energies", maxiter=1000, log=False
    ):
        """Run the optimization of LJ potentials.

        This function optimizes Lennard-Jones (LJ) epsilon and sigma values
        using either energy or forces as the objective. It supports an optional
        log scale transformation for the forces during optimization to handle
        large values.

        Parameters:
        ----------
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects with known energies or forces.
        minimized_by : str, optional, default="energies"
            Defines the objective for optimization. Choose between:
            - "energies": Optimize based on potential energies.
            - "forces": Optimize based on forces.
        maxiter : int, optional, default=100
            Maximum number of iterations allowed during the optimization process.
        log : bool, optional, default=False
            If True, apply a log scale transformation to forces to prevent large
            values from dominating the optimization.

        Returns:
        -------
        eps : dict
            Optimized epsilon values for each element.
        sigma : dict
            Optimized sigma values for each element.
        """
        logger.info(
            f"optimization is performed with {minimized_by}, with log_scale {log}"
        )
        opt_pars = [self.init_eps_dict[ele] for ele in self.element_list_sorted]
        opt_pars += [self.init_sigma_dict[ele] for ele in self.element_list_sorted]

        bounds = [(0, None)] * len(self.element_list_sorted) + [(1.5, 3.5)] * len(
            self.element_list_sorted
        )

        if minimized_by == "energies":
            self.build_atom_offset_ds(labelled_list_atoms)
            objective_func = self.objective_by_energy
            init_dict_atom_eng = self.chemipot
            opt_pars += [init_dict_atom_eng[ele] for ele in self.element_list_sorted]
            label = [atoms.get_potential_energy() for atoms in labelled_list_atoms]

            bounds += [(None, 0)] * len(self.element_list_sorted)

        elif minimized_by == "forces":
            objective_func = self.objective_by_forces
            label = [atoms.get_forces() for atoms in labelled_list_atoms]

        args = (self.element_list_sorted, labelled_list_atoms, log, label)
        res = minimize(
            objective_func,
            opt_pars,
            args=args,
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        parameters_optimized = res.x

        logger.info("optimization done")

        eps = {}
        sigma = {}
        for i, ele in enumerate(self.element_list_sorted):
            eps[ele] = parameters_optimized[i]
            sigma[ele] = parameters_optimized[i + len(self.element_list_sorted)]

        return eps, sigma

    @staticmethod
    def objective_by_energy(opt_paras, unique_atoms, labelled_list_atoms, log, label):
        """Objective function for minimizing energy differences.

        Minimizes the energy differences between labelled data and LJ-predicted data.

        Parameters
        ----------
        opt_paras : list of float
            List of optimized epsilon and sigma values.
        unique_atoms : list of str
            List of unique atomic symbols.
        labelled_list_atoms : list of ase.Atoms
            List of Atoms objects with known energies.

        Returns:
        -------
        float
            Root mean square error (RMSE) between labelled and LJ-calculated energies.
        """
        epsilon = 1e-8
        lj_engs = []
        # build dicts for lammps calculators
        eps = {}
        sigma = {}
        dict_atom_eng = {}

        # set labelled energy
        labelled_engs = np.array(label)
        if log:
            labelled_engs = np.sign(labelled_engs) * np.log(
                np.abs(labelled_engs) + epsilon
            )

        # set dict argss
        for i, ele in enumerate(unique_atoms):
            eps[ele] = float(opt_paras[i])
            sigma[ele] = float(opt_paras[i + 1 * len(unique_atoms)])
            dict_atom_eng[ele] = float(opt_paras[i + 2 * len(unique_atoms)])

        # lammps calculator
        lj_calc = LammpsLJBuilder().get_calculator(dict_eps=eps, dict_sigma=sigma)

        for atoms in labelled_list_atoms:
            _atoms = atoms.copy()
            _atoms.calc = lj_calc
            lj_eng = _atoms.get_potential_energy()
            # offset ~ chemical potential in this case.
            offset = np.array(
                [
                    dict_atom_eng[ele] * np.sum(atoms.symbols == ele)
                    for ele in dict_atom_eng.keys()
                ]
            ).sum()
            lj_eng += offset

            if log:
                lj_eng = np.sign(lj_eng) * np.log(np.abs(lj_eng) + epsilon)
            lj_engs.append(lj_eng)

        diff = np.array(labelled_engs) - np.array(lj_engs)
        rsme = np.linalg.norm(diff)

        print("(p^o^)p ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ (YoY)//")
        logger.info(f"objective : {rsme}")
        logger.info(f"epsilon : {eps}")
        logger.info(f"sigma : {sigma}")
        logging.info(f"chempot : {dict_atom_eng}")
        print("\\\(YoY) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ q(^o^q)")

        return rsme

    @staticmethod
    def objective_by_forces(opt_paras, unique_atoms, labelled_list_atoms, log, label):
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
        lj_forces = []
        epsilon = 1e-8

        # build dicts for lammps calculators
        eps = {}
        sigma = {}
        for i, ele in enumerate(unique_atoms):
            eps[ele] = opt_paras[i]
            sigma[ele] = opt_paras[i + len(unique_atoms)]

        # lammps calculator
        lj_calc = LammpsLJBuilder().get_calculator(dict_eps=eps, dict_sigma=sigma)

        # set labelled forces
        labelled_forces = np.array(label)
        if log:
            labelled_forces = np.sign(labelled_forces) * np.log(
                np.abs(labelled_forces) + epsilon
            )

        for atoms in labelled_list_atoms:
            _atoms = atoms.copy()
            _atoms.calc = lj_calc
            lj_forces_tmp = _atoms.get_forces()
            if log:
                lj_forces_tmp = np.sign(lj_forces_tmp) * np.log(
                    np.abs(lj_forces_tmp) + epsilon
                )
            lj_forces.append(lj_forces_tmp.tolist())

        lj_forces = np.array(lj_forces).reshape([-1, 1])[0]
        diff = labelled_forces - lj_forces

        rsme = np.linalg.norm(diff)

        print("(p^o^)p ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ (YoY)//")
        logger.info(f"objective : {rsme}")
        logger.info(f"epsilon : {eps}")
        logger.info(f"sigma : {sigma}")
        print("\\\(YoY) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ q(^o^q)")

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
            yield [
                i,
                sort(atoms).get_chemical_formula(),
                sort(atoms).symbols,
                atoms.get_potential_energy(),
                atoms,
            ]

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
            columns=["id", "chemical_formula", "compositions", "eng_eV", "AtomsObj"],
        )
        df.set_index("id", drop=True, inplace=True)

        def get_mins(df_dummy):
            min_id = df_dummy["eng_eV"].argmin()
            return df_dummy.iloc[int(min_id)]

        df_min = df.groupby(by="chemical_formula", as_index=False).apply(get_mins)
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
        regr.fit(train_x, train_y)
        self.chemipot = {
            self.element_list_sorted[i]: regr.coef_[i]
            for i in range(len(self.element_list_sorted))
        }
        logger.info("estimated rough chemical potentials")
        logger.info(f"{self.chemipot}")
