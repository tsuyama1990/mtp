"""Define Calculator for MTP."""

from ase.calculators.calculator import Calculator, all_changes

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
        self.results["forces"] = mtp_atoms.forces
        self.results["stress"] = mtp_atoms.stress
