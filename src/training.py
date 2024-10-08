"""Call training executables in MLIP-2."""

import shutil
import subprocess
from pathlib import Path

from ase import Atoms

from src import utils


class MtpTraining:
    """Class to handle training using MLIP-2 executables.

    This class manages the setup and execution of MLIP-2 training commands,
    modifying trainer input files, and running the training process with
    custom configuration files.

    Attributes:
    ----------
    command : str
        The command to run the training executable.
    cfg_path : Path
        Path to the CFG file.
    output_path : Path
        Path where the output will be saved.
    trainer_setting_path : Path, optional
        Path to the training setting file, if provided.
    exe_path : Path
        Path to the MLIP executable.
    trainer_path : Path
        Path to the generated or copied trainer file.
    num_para_cores : int
        Number of parallel cores for running the training.
    """

    def __init__(
        self,
        cfg_path: Path,
        output_path: Path,
        num_para_cores: int = 1,
        iter_limit: int = 100,
        trainer_setting_path: Path | None = None,
        base_trainer_name: str = "06.mtp",
        exe_path: Path | None = None,
    ):
        """Initialize the MtpTraining object.

        Parameters
        ----------
        cfg_path : Path
            Path to the CFG file.
        output_path : Path
            Path where the output will be saved.
        num_para_cores : int, optional
            Number of parallel cores to use for training, default is 1.
        iter_limit : int, optional
            Limit on the number of iterations, default is 100.
        trainer_setting_path : Path, optional
            Path to the training setting file, default is None.
        base_trainer_name : Str:
            The name of trainer file to use.
        exe_path : Path, optional
            Path to the MLIP executable, default is None.
        """
        self.cfg_path = cfg_path
        self.num_para_cores = num_para_cores
        self.output_path = output_path
        self.trainer_setting_path = trainer_setting_path

        output_path.parent.mkdir(exist_ok=True, parents=True)

        mlip_path = utils.find_mlip_dir()

        if exe_path is None:
            exe_path = mlip_path / "bin/mlp"
            self.exe_path = exe_path

        self.edit_trainer_file(base_trainer_name)

        self.command = self.build_command(iter_limit)

    def edit_trainer_file(self, base_trainer_name):
        """Edit or copy the trainer file.

        Modifies the original trainer file or copies a provided trainer file
        to the output directory and stores the path for further use.

        Parameters
        ----------
        base_trainer_name : Str:
            The name of trainer file to use.
        """
        if self.trainer_setting_path is None:
            trainer_path = self.output_path.parent / "trainer.mtp"
            edt = EditTrainerFile(self.cfg_path, trainer_path, base_trainer_name)
            edt.generate_modified_input()
        else:
            trainer_path = self.output_path.parent / "trainer.mtp"
            if trainer_path != self.trainer_setting_path:
                shutil.copy2(self.trainer_setting_path, trainer_path)

        self.trainer_path = trainer_path

    def build_command(
        self,
        iter_limit: int,
    ) -> str:
        """Build the command string to run the training executable.

        Parameters
        ----------
        iter_limit : int
            Limit on the number of iterations.

        Returns:
        -------
        str
            The command to run the training executable.
        """
        if self.num_para_cores == 1:
            _command = [
                str(self.exe_path),
                "train",
                str(self.trainer_path),
                str(self.cfg_path),
                f"--trained-pot-name={self.output_path}",
                f"--max-iter={iter_limit}",
            ]
        else:
            _command = [
                "mpirun -np",
                f"{self.num_para_cores}",
                str(self.exe_path),
                "train",
                str(self.trainer_path),
                str(self.cfg_path),
                f"--trained-pot-name={self.output_path}",
                f"--max-iter={iter_limit}",
            ]
        command = " ".join(_command)
        print("The command to execute:")
        print(command)
        return command

    def run(self):
        """Run the training executable with the constructed command."""
        subprocess.run(self.command, shell=True)


class EditTrainerFile:
    """Class to handle modifications of trainer input files for MLIP.

    This class manages generating, modifying, and editing trainer input files
    based on the dataset and MLIP executable.

    Attributes:
    ----------
    path_to_dataset : Path
        Path to the dataset used for training.
    output_file_path : Path
        Path where the modified trainer file will be saved.
    path_to_original_trainer : Path
        Path to the original trainer file template.
    exe_path : Path
        Path to the MLIP executable.
    """

    def __init__(
        self,
        path_to_dataset: Path,
        output_file_path: Path,
        base_trainer_name: str,
    ):
        """Initialize the EditTrainerFile object.

        Parameters:
        ----------
        path_to_dataset : Path
            Path to the dataset used for training.
        output_file_path : Path
            Path where the modified trainer file will be saved.
        base_trainer_name : str
            The trainer name for default settings.
        """
        self.path_to_dataset = path_to_dataset
        self.output_file_path = output_file_path

        mlip_path = utils.find_mlip_dir()
        self.path_to_original_trainer = (
            mlip_path / f"untrained_mtps/{base_trainer_name}"
        )
        self.exe_path = mlip_path / "bin/mlp"

    def get_mindist_command(self) -> str:
        """Construct the command to run the mindist executable.

        Returns:
        -------
        str
            The constructed command to calculate the minimum distance.
        """
        command = [
            str(self.exe_path),
            "mindist",
            str(self.path_to_dataset),
        ]
        return " ".join(command)

    def run_mindist(self) -> float:
        """Run the mindist executable and return the minimum distance.

        Returns:
        -------
        float
            The global minimum distance calculated by the mindist command.
        """
        cmd = self.get_mindist_command()
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)

        output = result.stdout.strip()  # Get rid of leading/trailing whitespaces
        dist = float(output[output.find(": ") + 2 :].strip())
        print(f"Output of mindist command was: {output}")
        return dist

    def get_unique_ele_num(self) -> int:
        """Get the number of unique elements in the dataset.

        Returns:
        -------
        int
            The number of unique elements in the dataset.
        """
        eles = utils.get_unique_element_from_cfg(self.path_to_dataset)
        print("The elements included in the cfg file", eles)
        atoms = Atoms(list(eles))
        return int(atoms.numbers.max())

    def generate_modified_input(self):
        """Generate the modified trainer input file."""
        with open(self.path_to_original_trainer) as f:
            ls = f.readlines()

        ls_modified = list(self.modify_input_byline(ls))

        with open(self.output_file_path, "w") as f:
            f.writelines(ls_modified)

    def modify_input_byline(self, lines):
        """Modify each line of the input file as needed.

        Yields:
        ------
        str
            A modified line of the trainer input file.
        """
        for line in lines:
            # modify species count
            if "species_count" in line:
                ele_num = self.get_unique_ele_num()
                line = f"species_count = {ele_num}\n"

            # modify min dist
            if "min_dist" in line:
                min_dist = self.run_mindist()
                line = line[: line.find("min_dist ")] + f"min_dist = {min_dist:.02f}\n"

            yield line


if __name__ == "__main__":
    current = Path(__file__).resolve()
    repo_path = current.parents[
        next(
            i + 1
            for i in range(len(current.parents))
            if current.parents[i].name == "src"
        )
    ]
    cfg_path = repo_path / "src/example/test.cfg"
    dataset_name = cfg_path.name.split(".")[0]
    output_path = repo_path / f"src/example/pot_{dataset_name}.mtp"

    mtp = MtpTraining(cfg_path, output_path)
    mtp.run()
