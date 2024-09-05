"""Optimize LJ potentials."""

import logging

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import sort
from scipy.optimize import minimize
from sklearn import linear_model

from src.ase_calculator_mtp import LammpsLJBuilder, LammpsLJBuilderLB

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LJTrainer:
    """A trainer for optimizing Lennard-Jones (LJ) potentials.

    This trainer does have the option of training using force, energy and hybrid.
    I recommend to use hybrid with e_weight < 0.1 for good accuracy.
    """

    def __init__(self, init_eps_dict=None, init_sigma_dict=None):
        """Initializes the LJTrainer class.

        Parameters:
        ----------
        init_eps_dict : dict, optional
            Initial dictionary of epsilon values for each element.
        init_sigma_dict : dict, optional
            Initial dictionary of sigma values for each element.
        """
        self.init_eps_dict = init_eps_dict
        self.init_sigma_dict = init_sigma_dict

    def build_init_pars(self, labelled_list_atoms):
        """Constructs initial parameters by labelled atomic systems.

        Parameters:
        ----------
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects containing atomic configurations.

        Returns:
        -------
        opt_pars : list
            List of initial parameters for LJ optimization, including epsilon, sigma,
            and atomic energies.
        """
        init_dict_atom_eng, unique_elements = self.build_atom_offset_ds(
            labelled_list_atoms
        )

        lj = LammpsLJBuilder()
        self.element_list_sorted, self.ele_num = lj.get_sorted_ele(unique_elements)
        self.possible_combo = lj.get_possible_combinations(self.element_list_sorted)

        # epsilon
        if self.init_eps_dict is not None:
            opt_pars = [self.init_eps_dict[pair] for pair in self.possible_combo]
        else:
            opt_pars = [
                -(init_dict_atom_eng[pair[0]] + init_dict_atom_eng[pair[1]]) / 4
                for pair in self.possible_combo
            ]
        # sigma
        if self.init_sigma_dict is not None:
            opt_pars += [self.init_sigma_dict[pair] for pair in self.possible_combo]
        else:
            opt_pars += [2.3 for _ in self.possible_combo]

        # atomic energy
        opt_pars += [init_dict_atom_eng[ele] / 4 for ele in self.element_list_sorted]

        return opt_pars

    def build_bounds(self, opt_pars):
        """Builds optimization bounds for LJ parameters.

        Parameters:
        ----------
        opt_pars : list
            List of initial parameters for optimization.

        Returns:
        -------
        bounds : list of tuple
            List of tuples defining the lower and upper bounds for epsilon, sigma,
            and atomic energy.
        """
        # bounds for epsilon
        bounds = [(0, None)] * len(self.possible_combo)
        # bounds for sigma
        bounds += [(1.5, 3.5)] * len(self.possible_combo)
        # bounds for atomic energy
        bounds += [
            (opt_pars[2 * len(self.possible_combo) + _id], 0)
            for _id in range(len(self.element_list_sorted))
        ]

        return bounds

    def run_minimize(
        self, labelled_list_atoms, minimized_by="e", maxiter=1000, log=False, **kwargs
    ):
        """Runs the optimization of LJ potentials.

        Parameters:
        ----------
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects with known energies or forces.
        minimized_by : str, optional
            Objective for optimization. Can be "e" (energy), "f" (forces), or "ef"
            (both). Default is "e".
        maxiter : int, optional
            Maximum number of iterations allowed during optimization. Default is 1000.
        log : bool, optional
            If True, apply log scale transformation to forces. Default is False.

        Returns:
        -------
        eps : dict
            Optimized epsilon values for each element.
        sigma : dict
            Optimized sigma values for each element.
        chempot : dict
            Optimized chemical potential values for each element.
        """
        logger.info(
            f"optimization is performed with {minimized_by}, with log_scale {log}."
        )
        opt_pars = self.build_init_pars(labelled_list_atoms)
        bounds = self.build_bounds(opt_pars)

        # Minimizing by energy
        if minimized_by == "e":
            objective_func = self.objective_by_energy
            label = [atoms.get_potential_energy() for atoms in labelled_list_atoms]
            logger.info(
                f"objective, epsilon:{self.possible_combo}, "
                f"sigma:{self.possible_combo}, "
                f"chempot:{self.possible_combo}"
            )
            args = (self.possible_combo, labelled_list_atoms, log, label)

        # Minimizing by forces
        elif minimized_by == "f":
            objective_func = self.objective_by_forces
            label = [atoms.get_forces() for atoms in labelled_list_atoms]
            logger.info(
                f"objective, epsilon:{self.possible_combo}, "
                f"sigma:{self.possible_combo}"
            )
            # Up to eps & sigma. No opt for atom eng.
            opt_pars = opt_pars[: 2 * len(opt_pars)]
            args = (self.possible_combo, labelled_list_atoms, log, label)

        # Minimizing by energy & forces
        elif minimized_by == "ef":
            objective_func = self.objective_by_energies_and_forces
            label = [atoms.get_potential_energy() for atoms in labelled_list_atoms]
            label += [atoms.get_forces() for atoms in labelled_list_atoms]
            logger.info(
                f"objective, epsilon:{self.possible_combo}, "
                f"sigma:{self.possible_combo}, "
                f"chempot:{self.possible_combo}"
            )
            e_weight = kwargs.get("e_weight", 0.05)
            args = (self.possible_combo, labelled_list_atoms, log, label, e_weight)

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
        chempot = {}
        _i = 0
        for i, pair in enumerate(self.possible_combo):
            eps[pair] = parameters_optimized[i]
            sigma[pair] = parameters_optimized[i + len(self.possible_combo)]
            if pair[0] == pair[1]:
                chempot[pair[0]] = parameters_optimized[
                    _i + len(self.possible_combo) * 2
                ]
                _i += 1

        return eps, sigma, chempot

    def objective_by_energy(self, opt_paras, pairs, labelled_list_atoms, log, label):
        """Objective function for minimizing energy differences.

        Minimizes the energy differences between labelled data and LJ-predicted data.

        Parameters:
        ----------
        opt_paras : list of float
            List of optimized epsilon and sigma values.
        pairs : list of str
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
        _i = 0
        for i, pair in enumerate(pairs):
            eps[pair] = float(opt_paras[i])
            sigma[pair] = float(opt_paras[i + 1 * len(pairs)])
            if pair[0] == pair[1]:
                dict_atom_eng[pair[0]] = float(opt_paras[_i + 2 * len(pairs)])
                _i += 1

        # lammps calculator
        lj_calc = LammpsLJBuilder().get_calculator(dict_eps=eps, dict_sigma=sigma)

        for atoms in labelled_list_atoms:
            _atoms = atoms.copy()
            _atoms.calc = lj_calc
            lj_eng = _atoms.get_potential_energy()
            # offset ~ chemical potential in this case.
            offset = np.array(
                [
                    dict_atom_eng[pair] * np.sum(atoms.symbols == pair[0])
                    for pair in dict_atom_eng.keys()
                    if pair[0] == pair[1]
                ]
            ).sum()
            lj_eng += offset

            if log:
                lj_eng = np.sign(lj_eng) * np.log(np.abs(lj_eng) + epsilon)
            lj_engs.append(lj_eng)

        diff = np.array(labelled_engs) - np.array(lj_engs)
        obj = np.sum(np.abs(diff))

        def set_digit(target_dict, num):
            return [round(float(val), num) for val in target_dict.values()]

        logger.info(
            f"{obj:.02f}, "
            f"{set_digit(eps, 4)}, "
            f"{set_digit(sigma, 2)}, "
            f"{set_digit(dict_atom_eng, 3)}"
        )

        return obj

    def objective_by_forces(self, opt_paras, pairs, labelled_list_atoms, log, label):
        """Objective function for LJ.

        to minimizing forces differences between labelled \
            data and LJ-predicted data.

        Parameters:
        ----------
        opt_paras : list of float
            Optimized epsilon and sigma values.
        pairs : list of str
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
        for i, pair in enumerate(pairs):
            eps[pair] = opt_paras[i]
            sigma[pair] = opt_paras[i + len(pairs)]

        # lammps calculator
        lj_calc = LammpsLJBuilder().get_calculator(dict_eps=eps, dict_sigma=sigma)

        # set labelled forces
        labelled_forces = np.array(label).reshape(1, -1)[0]
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

        lj_forces = np.array(lj_forces).reshape(1, -1)[0]
        diff = labelled_forces - lj_forces

        obj = np.sum(np.abs(diff))

        def set_digit(target_dict, num):
            return [round(float(val), num) for val in target_dict.values()]

        logger.info(f"{obj:.02f}, {set_digit(eps, 4)}, {set_digit(sigma, 2)}")

        return obj

    def objective_by_energies_and_forces(
        self, opt_paras, pairs, labelled_list_atoms, log, label, e_weight
    ):
        """Objective function for LJ.

        to minimizing forces differences between labelled \
            data and LJ-predicted data.

        Parameters:
        ----------
        opt_paras : list of float
            Optimized epsilon and sigma values.
        pairs : list of str
            List of unique atom symbols.
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects with known forces.

        Returns:
        -------
        float
            Root mean square error (RMSE) between labelled and LJ forces.
        """
        lj_engs = []
        lj_forces = []
        epsilon = 1e-8

        # build dicts for lammps calculators
        eps = {}
        sigma = {}
        dict_atom_eng = {}

        # set dict argss
        _i = 0
        for i, pair in enumerate(pairs):
            eps[pair] = float(opt_paras[i])
            sigma[pair] = float(opt_paras[i + 1 * len(pairs)])
            if pair[0] == pair[1]:
                dict_atom_eng[pair] = float(opt_paras[_i + 2 * len(pairs)])
                _i += 1

        # lammps calculator
        lj_calc = LammpsLJBuilder().get_calculator(dict_eps=eps, dict_sigma=sigma)

        # set labelled data
        labelled_engs = np.array(label[: len(labelled_list_atoms)])
        labelled_forces = np.array(label[len(labelled_list_atoms) :]).reshape(1, -1)[0]
        if log:
            labelled_engs = np.sign(labelled_engs) * np.log(
                np.abs(labelled_engs) + epsilon
            )
            labelled_forces = np.sign(labelled_forces) * np.log(
                np.abs(labelled_forces) + epsilon
            )

        for atoms in labelled_list_atoms:
            _atoms = atoms.copy()
            _atoms.calc = lj_calc

            # energy
            lj_eng = _atoms.get_potential_energy()
            # offset ~ chemical potential in this case.
            offset = np.array(
                [
                    dict_atom_eng[pair] * np.sum(atoms.symbols == pair[0])
                    for pair in dict_atom_eng.keys()
                    if pair[0] == pair[1]
                ]
            ).sum()
            lj_eng += offset

            if log:
                lj_eng = np.sign(lj_eng) * np.log(np.abs(lj_eng) + epsilon)
            lj_engs.append(lj_eng)

            # Forces
            lj_forces_tmp = _atoms.get_forces()
            if log:
                lj_forces_tmp = np.sign(lj_forces_tmp) * np.log(
                    np.abs(lj_forces_tmp) + epsilon
                )
            lj_forces.append(lj_forces_tmp.tolist())
        lj_forces = np.array(lj_forces).reshape(1, -1)[0]

        # Get Obj for Eng.
        diff_e = np.array(labelled_engs) - np.array(lj_engs)
        obj_e = np.sum(np.abs(diff_e))

        # Get Obj for frc
        diff_f = labelled_forces - lj_forces
        obj_f = np.sum(np.abs(diff_f))

        def set_digit(target_dict, num):
            return [round(float(val), num) for val in target_dict.values()]

        # Linear Combo of e & f
        obj = e_weight * obj_e + (1 - e_weight) * obj_f

        logger.info(
            f"{obj:.02f}, "
            f"{set_digit(eps, 4)}, "
            f"{set_digit(sigma, 2)}, "
            f"{set_digit(dict_atom_eng, 3)}"
        )

        return obj

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

    def get_atom_num(self, symbols, unique_elements):
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
        for ele in unique_elements:
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
        unique_elements = np.array(
            [list(set(i)) for i in df_min["compositions"].values]
        ).reshape(1, -1)[0]

        train_y = df_min["eng_eV"].values
        train_x = []
        for symbols in df_min["compositions"]:
            train_x.append(list(self.get_atom_num(symbols, unique_elements)))

        train_x = np.array(train_x)
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(train_x, train_y)
        chemipot = {
            unique_elements[i]: regr.coef_[i] for i in range(len(unique_elements))
        }
        logger.info("estimated rough chemical potentials")
        logger.info(f"{chemipot}")
        return chemipot, unique_elements


class LJTrainerLB(LJTrainer):
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

    def build_init_pars(self, labelled_list_atoms):
        """Constructs initial parameters by labelled atomic systems.

        Parameters:
        ----------
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects containing atomic configurations.

        Returns:
        -------
        opt_pars : list
            List of initial parameters for LJ optimization, including epsilon, sigma,
            and atomic energies.
        """
        init_dict_atom_eng, unique_elements = self.build_atom_offset_ds(
            labelled_list_atoms
        )

        lj = LammpsLJBuilderLB()
        self.element_list_sorted, self.ele_num = lj.get_sorted_ele(unique_elements)
        self.possible_combo = lj.get_possible_combinations(self.element_list_sorted)
        self.possible_combo = [i for i in self.possible_combo if i[0] == i[1]]

        # epsilon
        if self.init_eps_dict is not None:
            opt_pars = [
                self.init_eps_dict[pair]
                for pair in self.possible_combo
                if pair[0] == pair[1]
            ]
        else:
            opt_pars = [
                -(init_dict_atom_eng[pair[0]] + init_dict_atom_eng[pair[1]]) / 4
                for pair in self.possible_combo
                if pair[0] == pair[1]
            ]
        # sigma
        if self.init_sigma_dict is not None:
            opt_pars += [
                self.init_sigma_dict[pair]
                for pair in self.possible_combo
                if pair[0] == pair[1]
            ]
        else:
            opt_pars += [2.3 for pair in self.possible_combo if pair[0] == pair[1]]

        # atomic eng
        opt_pars += [init_dict_atom_eng[ele] / 4 for ele in self.element_list_sorted]

        return opt_pars

    def build_bounds(self, opt_pars):
        """Builds optimization bounds for LJ parameters.

        Parameters:
        ----------
        opt_pars : list
            List of initial parameters for optimization.

        Returns:
        -------
        bounds : list of tuple
            List of tuples defining the lower and upper bounds for epsilon, sigma,
            and atomic energy.
        """
        # bounds for epsilon
        bounds = [(0, None)] * len(self.element_list_sorted)
        # bounds for sigma
        bounds += [(1.5, 3.5)] * len(self.element_list_sorted)
        # bounds for atomic energy
        bounds += [
            (opt_pars[2 * len(self.element_list_sorted) + _id], 0)
            for _id in range(len(self.element_list_sorted))
        ]

        return bounds

    def run_minimize(
        self, labelled_list_atoms, minimized_by="e", maxiter=1000, log=False, **kwargs
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
            f"optimization is performed with {minimized_by}, with log_scale {log}."
        )
        opt_pars = self.build_init_pars(labelled_list_atoms)
        bounds = self.build_bounds(opt_pars)

        # Minimizing by energy
        if minimized_by == "e":
            objective_func = self.objective_by_energy
            label = [atoms.get_potential_energy() for atoms in labelled_list_atoms]
            logger.info(
                f"objective, epsilon:{self.element_list_sorted}, "
                f"sigma:{self.element_list_sorted}, "
                f"chempot:{self.element_list_sorted}"
            )
            args = (self.possible_combo, labelled_list_atoms, log, label)

        # Minimizing by forces
        elif minimized_by == "f":
            objective_func = self.objective_by_forces
            label = [atoms.get_forces() for atoms in labelled_list_atoms]
            logger.info(
                f"objective, epsilon:{self.element_list_sorted}, "
                f"sigma:{self.element_list_sorted}"
            )
            # Up to eps & sigma. No opt for atom eng.
            opt_pars = opt_pars[: 2 * len(opt_pars)]
            args = (self.possible_combo, labelled_list_atoms, log, label)

        # Minimizing by energy & forces
        elif minimized_by == "ef":
            objective_func = self.objective_by_energies_and_forces
            label = [atoms.get_potential_energy() for atoms in labelled_list_atoms]
            label += [atoms.get_forces() for atoms in labelled_list_atoms]
            logger.info(
                f"objective, epsilon:{self.element_list_sorted}, "
                f"sigma:{self.element_list_sorted}, "
                f"chempot:{self.element_list_sorted}"
            )
            e_weight = kwargs.get("e_weight", 0.05)
            args = (self.possible_combo, labelled_list_atoms, log, label, e_weight)

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
        chempot = {}
        _i = 0
        for i, pair in enumerate(self.element_list_sorted):
            eps[pair] = parameters_optimized[i]
            sigma[pair] = parameters_optimized[i + len(self.element_list_sorted)]
            chempot[pair] = parameters_optimized[i + len(self.element_list_sorted) * 2]

        return eps, sigma, chempot

    def objective_by_energy(self, opt_paras, pairs, labelled_list_atoms, log, label):
        """Objective function for minimizing energy differences.

        Minimizes the energy differences between labelled data and LJ-predicted data.

        Parameters
        ----------
        opt_paras : list of float
            List of optimized epsilon and sigma values.
        pairs : list of str
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
        for i, pair in enumerate(pairs):
            eps[pair] = float(opt_paras[i])
            sigma[pair] = float(opt_paras[i + 1 * len(pairs)])
            dict_atom_eng[pair] = float(opt_paras[i + 2 * len(pairs)])

        # lammps calculator
        lj_calc = LammpsLJBuilderLB().get_calculator(dict_eps=eps, dict_sigma=sigma)

        for atoms in labelled_list_atoms:
            _atoms = atoms.copy()
            _atoms.calc = lj_calc
            lj_eng = _atoms.get_potential_energy()
            # offset ~ chemical potential in this case.
            offset = np.array(
                [
                    dict_atom_eng[pair] * np.sum(atoms.symbols == pair[0])
                    for pair in dict_atom_eng.keys()
                ]
            ).sum()
            lj_eng += offset

            if log:
                lj_eng = np.sign(lj_eng) * np.log(np.abs(lj_eng) + epsilon)
            lj_engs.append(lj_eng)

        diff = np.array(labelled_engs) - np.array(lj_engs)
        obj = np.sum(np.abs(diff))

        def set_digit(target_dict, num):
            return [round(float(val), num) for val in target_dict.values()]

        logger.info(
            f"{obj:.02f}, "
            f"{set_digit(eps, 4)}, "
            f"{set_digit(sigma, 2)}, "
            f"{set_digit(dict_atom_eng, 3)}"
        )

        return obj

    def objective_by_forces(self, opt_paras, pairs, labelled_list_atoms, log, label):
        """Objective function for LJ.

        to minimizing forces differences between labelled \
            data and LJ-predicted data.

        Parameters:
        ----------
        opt_paras : list of float
            Optimized epsilon and sigma values.
        pairs : list of str
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
        for i, pair in enumerate(pairs):
            eps[pair] = opt_paras[i]
            sigma[pair] = opt_paras[i + len(pairs)]

        # lammps calculator
        lj_calc = LammpsLJBuilderLB().get_calculator(dict_eps=eps, dict_sigma=sigma)

        # set labelled forces
        labelled_forces = np.array(label).reshape(1, -1)[0]
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

        lj_forces = np.array(lj_forces).reshape(1, -1)[0]
        diff = labelled_forces - lj_forces

        obj = np.sum(np.abs(diff))

        def set_digit(target_dict, num):
            return [round(float(val), num) for val in target_dict.values()]

        logger.info(f"{obj:.02f}, {set_digit(eps, 4)}, {set_digit(sigma, 2)}")

        return obj

    def objective_by_energies_and_forces(
        self, opt_paras, pairs, labelled_list_atoms, log, label, e_weight
    ):
        """Objective function for LJ.

        to minimizing forces differences between labelled \
            data and LJ-predicted data.

        Parameters:
        ----------
        opt_paras : list of float
            Optimized epsilon and sigma values.
        pairs : list of str
            List of unique atom symbols.
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects with known forces.

        Returns:
        -------
        float
            Root mean square error (RMSE) between labelled and LJ forces.
        """
        lj_engs = []
        lj_forces = []
        epsilon = 1e-8

        # build dicts for lammps calculators
        eps = {}
        sigma = {}
        dict_atom_eng = {}

        # set dict argss
        for i, pair in enumerate(pairs):
            eps[pair] = float(opt_paras[i])
            sigma[pair] = float(opt_paras[i + 1 * len(pairs)])
            dict_atom_eng[pair] = float(opt_paras[i + 2 * len(pairs)])

        # lammps calculator
        lj_calc = LammpsLJBuilderLB().get_calculator(dict_eps=eps, dict_sigma=sigma)

        # set labelled data
        labelled_engs = np.array(label[: len(labelled_list_atoms)])
        labelled_forces = np.array(label[len(labelled_list_atoms) :]).reshape(1, -1)[0]
        if log:
            labelled_engs = np.sign(labelled_engs) * np.log(
                np.abs(labelled_engs) + epsilon
            )
            labelled_forces = np.sign(labelled_forces) * np.log(
                np.abs(labelled_forces) + epsilon
            )

        for atoms in labelled_list_atoms:
            _atoms = atoms.copy()
            _atoms.calc = lj_calc

            # energy
            lj_eng = _atoms.get_potential_energy()
            # offset ~ chemical potential in this case.
            offset = np.array(
                [
                    dict_atom_eng[pair] * np.sum(atoms.symbols == pair[0])
                    for pair in dict_atom_eng.keys()
                ]
            ).sum()
            lj_eng += offset

            if log:
                lj_eng = np.sign(lj_eng) * np.log(np.abs(lj_eng) + epsilon)
            lj_engs.append(lj_eng)

            # Forces
            lj_forces_tmp = _atoms.get_forces()
            if log:
                lj_forces_tmp = np.sign(lj_forces_tmp) * np.log(
                    np.abs(lj_forces_tmp) + epsilon
                )
            lj_forces.append(lj_forces_tmp.tolist())
        lj_forces = np.array(lj_forces).reshape(1, -1)[0]

        # Get Obj for Eng.
        diff_e = np.array(labelled_engs) - np.array(lj_engs)
        obj_e = np.sum(np.abs(diff_e))

        # Get Obj for frc
        diff_f = labelled_forces - lj_forces
        obj_f = np.sum(np.abs(diff_f))

        def set_digit(target_dict, num):
            return [round(float(val), num) for val in target_dict.values()]

        # Linear Combo of e & f
        obj = e_weight * obj_e + (1 - e_weight) * obj_f

        logger.info(
            f"{obj:.02f}, "
            f"{set_digit(eps, 4)}, "
            f"{set_digit(sigma, 2)}, "
            f"{set_digit(dict_atom_eng, 3)}"
        )

        return obj
