"""Call training executables in MLIP-2."""

import subprocess
from pathlib import Path


class MtpTraining:
    """Class to handle training using MLIP-2 executables.

    Attributes:
    ----------
    command : str
        The command to run the training executable.
    """

    def __init__(
        self,
        setting_path: Path,
        exe_path: Path,
        cfg_path: Path,
        output_path: Path,
        iter_limit: int = 100,
        al_mode: str = "nbh",
    ):
        """Initialize the MtpTraining object.

        Parameters
        ----------
        setting_path : Path
            Path to the training setting file.
        exe_path : Path
            Path to the MLIP executable.
        cfg_path : Path
            Path to the CFG file.
        output_path : Path
            Path where the output will be saved.
        iter_limit : int, optional
            Limit on the number of iterations, by default 100.
        al_mode : str, optional
            Active learning mode, by default "nbh".
        """
        output_path.parent.mkdir(exist_ok=True, parents=True)
        self.command = self.build_command(
            setting_path, exe_path, cfg_path, output_path, iter_limit, al_mode
        )

    def build_command(
        self,
        setting_path: Path,
        exe_path: Path,
        cfg_path: Path,
        output_path: Path,
        iter_limit: int,
        al_mode: str,
    ) -> str:
        """Build the command string to run the training executable.

        Parameters
        ----------
        setting_path : Path
            Path to the training setting file.
        exe_path : Path
            Path to the MLIP executable.
        cfg_path : Path
            Path to the CFG file.
        output_path : Path
            Path where the output will be saved.
        iter_limit : int
            Limit on the number of iterations.
        al_mode : str
            Active learning mode.

        Returns:
        -------
        str
            The command to run the training executable.
        """
        command = [
            str(exe_path),
            "train",
            str(setting_path),
            str(cfg_path),
            f"--save_to={output_path}",
            f"--iteration_limit={iter_limit}",
            f"--al_mode={al_mode}",
        ]
        return " ".join(command)

    def run(self):
        """Run the training executable with the constructed command."""
        print(self.command)
        subprocess.run(self.command, shell=True)


if __name__ == "__main__":
    current = Path(__file__).resolve()
    repo_path = current.parents[
        next(
            i + 1
            for i in range(len(current.parents))
            if current.parents[i].name == "src"
        )
    ]
    mtp_path = repo_path / "mlip_2"

    setting_path = mtp_path / "MTP_templates/06.almtp"
    exe_path = mtp_path / "bin/mlp"
    cfg_path = repo_path / "src/example/test.cfg"
    dataset_name = cfg_path.name.split(".")[0]
    output_path = repo_path / f"src/example/pot_{dataset_name}.almtp"

    mtp = MtpTraining(setting_path, exe_path, cfg_path, output_path)
    mtp.run()
