"""Optimize LJ potentials."""

import logging

import numpy as np
import pandas as pd
import yaml
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

    def __init__(
        self,
        init_eps_dict={},
        init_sigma_dict={},
        init_chempot_dict={},
        fix_pair_lists=[],
    ):
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
        self.init_chempot_dict = init_chempot_dict
        self.fix_pair_lists = fix_pair_lists

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
        # if init_eps_dict contains the value for specific pair, register.
        # Otherwise the initial value calculated by dft is registered.
        # The pair energy is estimated by dividing the order of 10.
        # Correspondign to the number of neighbors
        self.eps = {}
        for pair in self.possible_combo:
            self.eps[pair] = self.init_eps_dict.get(
                pair, -(init_dict_atom_eng[pair[0]] + init_dict_atom_eng[pair[1]]) / 15
            )

        opt_pars = [
            self.eps[pair]
            for pair in self.possible_combo
            if pair not in self.fix_pair_lists
        ]

        # sigma
        # Same as for eps
        self.sigma = {}
        for pair in self.possible_combo:
            self.sigma[pair] = self.init_sigma_dict.get(pair, 2.4)

        opt_pars += [
            self.sigma[pair]
            for pair in self.possible_combo
            if pair not in self.fix_pair_lists
        ]

        # atomic energy
        self.chempot = {}
        for ele in self.element_list_sorted:
            self.chempot[ele] = self.init_chempot_dict.get(
                ele, init_dict_atom_eng[ele] / 2
            )

        opt_pars += [self.chempot[ele] for ele in self.element_list_sorted]

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
        opt_pairs = [
            pair for pair in self.possible_combo if pair not in self.fix_pair_lists
        ]
        # bounds for epsilon1
        bounds = [(0, None)] * len(opt_pairs)
        # bounds for sigma11
        bounds += [(1.5, 3.5)] * len(opt_pairs)
        # bounds for atomic energy
        bounds += [
            (opt_pars[2 * len(opt_pairs) + _id], 0)
            for _id in range(len(self.element_list_sorted))
        ]

        return bounds

    def save_file(self, save_to_file):
        """Save the optimized value on save_to_file argment.

        Parameters:
        -------
        save_to_file
            If not None, eps, sigma, and chemical potential values are saved,
            as a yaml file on this path.
        """
        if save_to_file is not None:
            # Convert tuple keys to string format for human readability
            eps_str_keys = {f"{k[0]}-{k[1]}": float(v) for k, v in self.eps.items()}
            sigma_str_keys = {f"{k[0]}-{k[1]}": float(v) for k, v in self.sigma.items()}
            chempot_str_keys = {k: float(v) for k, v in self.chempot.items()}

            data = {
                "eps": eps_str_keys,
                "sigma": sigma_str_keys,
                "chempot": chempot_str_keys,
                "element_id": self.ele_num,
            }

            with open(save_to_file, "w") as yamlfile:
                yaml.dump(data, yamlfile, default_flow_style=False)

    def run_minimize(
        self,
        labelled_list_atoms,
        maxiter=1000,
        log=False,
        save_to_file=None,
        e_weight=0.05,
    ):
        """Runs the optimization of LJ potentials.

        Parameters:
        ----------
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects with known energies or forces.
        maxiter : int, optional
            Maximum number of iterations allowed during optimization. Default is 1000.
        log : bool, optional
            If True, apply log scale transformation to forces. Default is False.
        save_to_file
            If not None, eps, sigma, and chemical potential values are saved,
            as a yaml file on this path.

        Returns:
        -------
        eps : dict
            Optimized epsilon values for each element.
        sigma : dict
            Optimized sigma values for each element.
        chempot : dict
            Optimized chemical potential values for each element.
        """
        opt_pars = self.build_init_pars(labelled_list_atoms)
        bounds = self.build_bounds(opt_pars)

        # Minimizing by energy & forces
        objective_func = self.objective
        label = [atoms.get_potential_energy() for atoms in labelled_list_atoms]
        label += [atoms.get_forces() for atoms in labelled_list_atoms]
        logger.info(
            f"objective: [e, f], "
            f"epsilon:{self.possible_combo}, "
            f"sigma:{self.possible_combo}, "
            f"chempot:{self.possible_combo}"
        )
        args = (labelled_list_atoms, log, label, e_weight)

        res = minimize(
            objective_func,
            opt_pars,
            args=args,
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        parameters_optimized = res.x
        parameters_optimized = [float(i) for i in parameters_optimized]

        logger.info("optimization done")

        opt_pairs = [
            pair for pair in self.possible_combo if pair not in self.fix_pair_lists
        ]
        _i = 0
        for i, pair in enumerate(opt_pairs):
            self.eps[pair] = parameters_optimized[i]
            self.sigma[pair] = parameters_optimized[i + len(opt_pairs)]
            if pair[0] == pair[1]:
                self.chempot[pair[0]] = parameters_optimized[_i + len(opt_pairs) * 2]
                _i += 1

        self.save_file(save_to_file)

        return self.eps, self.sigma, self.chempot

    def objective(self, opt_paras, labelled_list_atoms, log, label, e_weight):
        """Objective function for LJ.

        to minimizing forces differences between labelled \
            data and LJ-predicted data.

        Parameters:
        ----------
        opt_paras : list of float
            Optimized epsilon and sigma values.
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects with known forces.
        log : bool
            Bool to use log scale for objective func.
        label : list
            List composed of energies and forces.
        e_weight : float
            Weight value of energy to forces for penalty.

        Returns:
        -------
        float
            Root mean square error (RMSE) between labelled and LJ forces.
        """
        lj_engs = []
        lj_forces = []
        epsilon = 1e-8

        # build dicts for lammps calculators

        # set dict argss
        opt_pairs = [
            pair for pair in self.possible_combo if pair not in self.fix_pair_lists
        ]
        _i = 0
        for i, pair in enumerate(opt_pairs):
            self.eps[pair] = float(opt_paras[i])
            self.sigma[pair] = float(opt_paras[i + 1 * len(opt_pairs)])
            if pair[0] == pair[1]:
                self.chempot[pair] = float(opt_paras[_i + 2 * len(opt_pairs)])
                _i += 1

        # lammps calculator
        lj_calc = LammpsLJBuilder().get_calculator(
            dict_eps=self.eps, dict_sigma=self.sigma
        )

        # set labelled data
        labelled_engs = np.array(label[: len(labelled_list_atoms)])

        tmp_forces = label[len(labelled_list_atoms) :]
        labelled_forces = np.array([item for sublist in tmp_forces for item in sublist])

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
                    self.chempot[pair] * np.sum(atoms.symbols == pair[0])
                    for pair in self.chempot.keys()
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

        lj_forces = np.array([item for sublist in lj_forces for item in sublist])

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
        obj_component = [round(float(i), 2) for i in [obj_e, obj_f]]

        logger.info(
            f"{obj:.02f}, {obj_component}, "
            f"{set_digit(self.eps, 4)}, "
            f"{set_digit(self.sigma, 2)}, "
            f"{set_digit(self.chempot, 3)}"
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

        lj = LammpsLJBuilder()
        self.element_list_sorted, self.ele_num = lj.get_sorted_ele(unique_elements)
        self.possible_combo = lj.get_possible_combinations(self.element_list_sorted)

        # epsilon
        # if init_eps_dict contains the value for specific pair, register.
        # Otherwise the initial value calculated by dft is registered.
        # The pair energy is estimated by dividing the order of 10.
        # Correspondign to the number of neighbors
        self.eps = {}
        for pair in self.possible_combo:
            self.eps[pair] = self.init_eps_dict.get(
                pair, -(init_dict_atom_eng[pair[0]] + init_dict_atom_eng[pair[1]]) / 15
            )

        opt_pars = [
            self.eps[pair]
            for pair in self.possible_combo
            if (pair not in self.fix_pair_lists) & (pair[0] == pair[1])
        ]

        # sigma
        # Same as for eps
        self.sigma = {}
        for pair in self.possible_combo:
            self.sigma[pair] = self.init_sigma_dict.get(pair, 2.4)

        opt_pars += [
            self.sigma[pair]
            for pair in self.possible_combo
            if (pair not in self.fix_pair_lists) & (pair[0] == pair[1])
        ]

        # atomic energy
        self.chempot = {}
        for ele in self.element_list_sorted:
            self.chempot[ele] = self.init_chempot_dict.get(
                ele, init_dict_atom_eng[ele] / 2
            )

        opt_pars += [self.chempot[ele] for ele in self.element_list_sorted]

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
        opt_pairs = [
            pair
            for pair in self.possible_combo
            if (pair not in self.fix_pair_lists) & (pair[0] == pair[1])
        ]
        # bounds for epsilon1
        bounds = [(0, None)] * len(opt_pairs)
        # bounds for sigma11
        bounds += [(1.5, 3.5)] * len(opt_pairs)
        # bounds for atomic energy
        bounds += [
            (opt_pars[2 * len(opt_pairs) + _id], 0)
            for _id in range(len(self.element_list_sorted))
        ]

    def run_minimize(
        self,
        labelled_list_atoms,
        maxiter=1000,
        log=False,
        save_to_file=None,
        e_weight=0.05,
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
        opt_pars = self.build_init_pars(labelled_list_atoms)
        bounds = self.build_bounds(opt_pars)

        # Minimizing by energy & forces
        objective_func = self.objective_by_energies_and_forces
        label = [atoms.get_potential_energy() for atoms in labelled_list_atoms]
        label += [atoms.get_forces() for atoms in labelled_list_atoms]
        logger.info(
            f"objective, epsilon:{self.element_list_sorted}, "
            f"sigma:{self.element_list_sorted}, "
            f"chempot:{self.element_list_sorted}"
        )
        args = (labelled_list_atoms, log, label, e_weight)

        res = minimize(
            objective_func,
            opt_pars,
            args=args,
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        parameters_optimized = res.x

        logger.info("optimization done")

        opt_pairs = [
            pair
            for pair in self.possible_combo
            if (pair not in self.fix_pair_lists) & (pair[0] == pair[1])
        ]
        for i, pair in enumerate(opt_pairs):
            self.eps[pair] = parameters_optimized[i]
            self.sigma[pair] = parameters_optimized[i + len(opt_pairs)]
            self.chempot[pair[0]] = parameters_optimized[i + len(opt_pairs) * 2]

        self.save_file(save_to_file)

        return self.eps, self.sigma, self.chempot

    def objective(self, opt_paras, labelled_list_atoms, log, label, e_weight):
        """Objective function for LJ.

        to minimizing forces differences between labelled \
            data and LJ-predicted data.

        Parameters:
        ----------
        opt_paras : list of float
            Optimized epsilon and sigma values.
        labelled_list_atoms : list of ase.Atoms
            List of labelled Atoms objects with known forces.
        log : bool
            Bool to use log scale for objective func.
        label : list
            List composed of energies and forces.
        e_weight : float
            Weight value of energy to forces for penalty.

        Returns:
        -------
        float
            Root mean square error (RMSE) between labelled and LJ forces.
        """
        lj_engs = []
        lj_forces = []
        epsilon = 1e-8

        # set dict argss
        opt_pairs = [
            pair
            for pair in self.possible_combo
            if (pair not in self.fix_pair_lists) & (pair[0] == pair[1])
        ]
        # set dict argss
        for i, pair in enumerate(opt_pairs):
            self.eps[pair] = float(opt_paras[i])
            self.sigma[pair] = float(opt_paras[i + 1 * len(opt_pairs)])
            self.dict_atom_eng[pair] = float(opt_paras[i + 2 * len(opt_pairs)])

        # lammps calculator
        lj_calc = LammpsLJBuilderLB().get_calculator(
            dict_eps=self.eps, dict_sigma=self.sigma
        )

        # set labelled data
        labelled_engs = np.array(label[: len(labelled_list_atoms)])
        tmp_forces = label[len(labelled_list_atoms) :]
        labelled_forces = np.array([item for sublist in tmp_forces for item in sublist])
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
                    self.chempot[pair] * np.sum(atoms.symbols == pair[0])
                    for pair in self.chempot.keys()
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

        lj_forces = np.array([item for sublist in lj_forces for item in sublist])

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
        obj_component = [round(float(i), 2) for i in [obj_e, obj_f]]

        logger.info(
            f"{obj:.02f}, {obj_component}, "
            f"{set_digit(self.eps, 4)}, "
            f"{set_digit(self.sigma, 2)}, "
            f"{set_digit(self.chempot, 3)}"
        )

        return obj
