"""Run a trainer separately and publish its completed checkpoint."""

import os
import re
import subprocess
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

PROGRESS_LINE = re.compile(r"^\s*(\d+)/\d+\s")


def trainer_command(args: Namespace, trainer: Path) -> list[str]:
    command = [sys.executable, "-X", "utf8", "-u", str(trainer)]
    for name, value in vars(args).items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif isinstance(value, list):
            for item in value:
                command.extend((option, str(item)))
        elif value is not None:
            command.extend((option, str(value)))
    return command


def file_revision(path: Path) -> tuple[int, int, int, int] | None:
    try:
        state = path.lstat()
    except FileNotFoundError:
        return None
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns


@dataclass(frozen=True)
class TrainingUpdate:
    message: str
    iteration: int | None = None
    finished: bool = False
    error: bool = False


class TrainingJob:
    def __init__(self, trainer: Path, options: Namespace, destination: Path) -> None:
        self.destination = destination
        self._revision = file_revision(destination)
        self._workdir = TemporaryDirectory(
            prefix=".studio-training-",
            dir=destination.parent,
            ignore_cleanup_errors=True,
        )
        directory = Path(self._workdir.name)
        self._checkpoint = directory / "model.pt"
        self._log_path = directory / "training.log"
        command = trainer_command(
            Namespace(**(vars(options) | {"output": self._checkpoint})), trainer
        )
        try:
            with self._log_path.open("wb") as log:
                self._process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
        except OSError:
            self._workdir.cleanup()
            raise

    def poll(self) -> TrainingUpdate:
        result = self._process.poll()
        try:
            with self._log_path.open("rb") as log:
                size = log.seek(0, os.SEEK_END)
                log.seek(max(0, size - 4096))
                lines = log.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        if result is None:
            message = lines[-1].strip() if lines else ""
            match = PROGRESS_LINE.match(message)
            return TrainingUpdate(message, int(match[1]) if match else None)

        try:
            if result != 0:
                raise ValueError(lines[-1] if lines else f"Trainer exited with code {result}.")
            if file_revision(self.destination) != self._revision:
                raise ValueError(
                    "The destination changed during training. Choose another filename."
                )
            os.replace(self._checkpoint, self.destination)
        except (OSError, ValueError) as error:
            return TrainingUpdate(str(error), finished=True, error=True)
        finally:
            self._workdir.cleanup()
        return TrainingUpdate("Training complete.", finished=True)

    def cancel(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            self._process.wait()
        self._workdir.cleanup()
