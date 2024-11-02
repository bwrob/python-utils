"""Python startup script."""

import os
import subprocess
import time
from collections.abc import Callable, Generator
from enum import Enum
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")
Task = tuple[T, int]
TaskListOptionalDelay = list[Task[T] | T]
TaskList = list[Task[T]]

NAME_EXCLUDES = ("$", "tmp")
EXT_EXCLUDES = ("exe",)
DEFAULT_DELAY_SECONDS = 4
FILES_IN_TREE_PATTERN = r"**\*.*"

logger = getLogger(__name__)


class Program(Enum):
    """Types of programs."""

    POWERSHELL = Path(r"C:\windows\system32\windowspowershell\v1.0\powershell.exe")
    NOTEPAD = Path(r"C:\Program Files\Notepad++\notepad++.exe")
    FIREFOX = Path(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Firefox.lnk")


def filter_excluded(
    path: Path,
) -> bool:
    """Filter path based on name and extension exclude lists.

    Args:
    ----
        path: Path to filter.

    """
    return (path.stem not in NAME_EXCLUDES) and (path.suffix not in EXT_EXCLUDES)


def start_process(
    *,
    name: str,
    path: Path,
    delay: int = DEFAULT_DELAY_SECONDS,
) -> None:
    """Given a path, starts the target.

    Behavior:
        * Minimizes all windows.
        * Depending on the path target:
            * executable files are run,
            * content files are opened with system default program,
            * folders are opened with system explorer.

    Args:
    ----
        name: Name of the process to start.
        path: Path to the target.
        delay: Time to wait after starting the process.

    """
    if path.is_dir():
        logger.info("Opening folder %s", name)

    if path.suffix in (".exe", ".lnk"):
        logger.info("Running app %s", name)
    else:
        logger.info("Opening file %s", name)

    os.startfile(path)  # noqa: S606
    time.sleep(delay)


def run_command(
    command: str,
    delay: int,
) -> None:
    """Run a powershell command.

    Args:
    ----
        command: Command to run.
        delay: Time to wait after starting the process.

    """
    _ = subprocess.call(  # noqa: S603
        f"powershell.exe {command}",
        shell=False,
    )
    time.sleep(delay)


def path_files(
    directory_tasks: TaskList[Path],
) -> Generator[tuple[Path, int], None, None]:
    """Generate all files in the work folder that are not excluded.

    Yields the folder path at beginning of the generator.

    Args:
    ----
        directory_tasks: List of tasks with paths to work folders.

    """
    for folder, delay in directory_tasks:
        yield folder, delay

        all_files = folder.glob(FILES_IN_TREE_PATTERN)
        filtered = filter(filter_excluded, all_files)

        for file in filtered:
            yield file, delay


def with_optional_delay(
    task_worker: Callable[[TaskList[T]], None],
) -> Callable[[TaskListOptionalDelay[T]], None]:
    """Add default delay to all non-tuple items.

    Args:
    ----
        task_worker: A function that takes a list of tasks.

    """

    @wraps(task_worker)
    def task_defaulted_worker(task_list: TaskListOptionalDelay[T]) -> None:
        """Add a default delay to tasks in a task list if no delay is specified.

        Args:
        ----
            task_list: A list of tasks with optional delays.

        """
        tasks_with_defaulted_delays: TaskList[T] = [
            item if isinstance(item, tuple) else (item, DEFAULT_DELAY_SECONDS)
            for item in task_list
        ]
        return task_worker(tasks_with_defaulted_delays)

    return task_defaulted_worker


@with_optional_delay
def start_programs(
    programs: TaskList[Program],
) -> None:
    """Start listed programs.

    Args:
    ----
        programs: List of programs to start.
            Can be a string or a tuple. If a tuple is given, the first
            element is the name, the second is the delay.

    """
    for program, delay in programs:
        start_process(
            name=program.name,
            path=program.value,
            delay=delay,
        )


@with_optional_delay
def start_work_files(
    directory_tasks: TaskList[Path],
) -> None:
    """Start all  files in the work folders.

    Args:
    ----
        directory_tasks: List of tasks with paths to work folders.

    """
    for path, delay in path_files(directory_tasks):
        start_process(
            name=path.name,
            path=path,
            delay=delay,
        )


@with_optional_delay
def run_commands(
    commands: TaskList[str],
) -> None:
    """Run all commands in the command list.

    Args:
    ----
        commands: List of commands to run.

    """
    for command, delay in commands:
        run_command(command, delay)


def run_startup_script() -> None:
    """Run the startup script."""
    logger.info(f" Welcome {os.getlogin()}! ".center(40, "*"))
    start_programs(
        [
            Program.POWERSHELL,
            Program.NOTEPAD,
            (Program.FIREFOX, 8),
        ],
    )
    start_work_files([Path.home() / ".temp"])
    run_commands(['Write-Output "test"'])


if __name__ == "__main__":
    run_startup_script()
