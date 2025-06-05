"""Live progress manager using multiprocessing and rich for visual feedback.

This module demonstrates how to manage and display live progress bars for multiple
concurrent tasks using Python's multiprocessing and the rich library.
"""

import multiprocessing as mp
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from multiprocessing import Manager, Pool
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Queue as _ManagerQueue
from multiprocessing.synchronize import Lock as ManagerLock
from typing import override

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

# --- Constants ---
# Task Simulation Constants
INITIAL_SLEEP_MIN: float = 0.1
INITIAL_SLEEP_MAX: float = 0.5
STEP_SLEEP_MIN: float = 0.01
STEP_SLEEP_MAX: float = 0.03

# Progress Management Constants
LOCK_ACQUISITION_RETRIES: int = 10
LOCK_ACQUISITION_TIMEOUT: float = 0.1
LOCK_RETRY_SLEEP: float = 0.05
QUEUE_GET_TIMEOUT: float = 0.1

# Rich Progress Display Constants
PROGRESS_REFRESH_RATE: int = 10
DEFAULT_WAITING_TASK_DESCRIPTION: str = "[dim]Waiting for task[/dim]"
OVERALL_PROGRESS_DESCRIPTION: str = "[green]Overall Progress"
WARNING_NO_SLOT_MESSAGE: str = (
    "Warning: Task {} could not acquire a display slot. Total sleep: {:.2f}s"
)
TASK_FINISHED_MESSAGE: str = "Task finished. Total sleep: {:.2f}s"
ALL_PROCESSING_COMPLETE_MESSAGE: str = "\nAll processing and display complete."
TOTAL_SLEEP_TIMES_HEADER: str = "\nTotal sleep times for all tasks:"
TASK_SLEEP_TIME_FORMAT: str = "Task {}: {:.2f}s"
AVERAGE_SLEEP_TIME_FORMAT: str = "\nAverage sleep time: {:.2f}s"


NUM_TASKS: int = 40
MIN_STEPS: int = 50
MAX_STEPS: int = 200
NUM_PROCESSES: int = 10

# --- Enums and Dataclasses for Shared Resources and Queue Messages ---


class ProgressMessageType(StrEnum):
    """Enum to define the type of progress message."""

    START = auto()
    UPDATE = auto()
    FINISH = auto()
    LOG = auto()


@dataclass
class ProgressMessage[T]:
    """Dataclass to represent a message sent through the progress queue."""

    type: ProgressMessageType
    task_id: TaskID | None = None
    description: str | None = None
    total: int | None = None
    advance: int | None = None
    message: str | None = None
    result_data: T | None = None
    start_time: float | None = None


type ManagerQueue[T] = _ManagerQueue[ProgressMessage[T]]


@dataclass
class SharedResources[T]:
    """Dataclass to hold shared resources for multiprocessing tasks."""

    task_ids: list[int]
    lock: ManagerLock
    queue: ManagerQueue[T]


# --- Custom Rich Progress Column ---
class TaskElapsedTimeColumn(ProgressColumn):
    """A custom progress column.

    To display the elapsed time for each individual task and  the overall progress task.
    """

    def __init__(
        self,
        task_start_times: dict[TaskID, float],
        overall_task_id: TaskID,
    ) -> None:
        """Initialize the TaskElapsedTimeColumn with start times and overall task ID."""
        super().__init__()
        self.task_start_times: dict[TaskID, float] = task_start_times
        self.overall_task_id: TaskID = overall_task_id

    @override
    def render(self, task: Task) -> Text:
        """Render the time elapsed for the given task."""
        if task.id == self.overall_task_id:
            # For the overall task, use rich's built-in elapsed time
            elapsed_time = task.elapsed
        elif task.id in self.task_start_times:
            # For individual worker tasks, use our custom recorded start time
            elapsed_time = time.monotonic() - self.task_start_times[task.id]
        else:
            # If start time not yet recorded or task finished, show placeholder
            return Text("-:--:--")

        return Text(str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))


def _perform_task_work(
    task_id_num: int,
    total_steps: int,
    rich_task_id: TaskID | None,
    progress_queue: ManagerQueue[ProgressMessage[float]],
) -> float:
    total_sleep_time = 0.0

    initial_sleep = random.uniform(INITIAL_SLEEP_MIN, INITIAL_SLEEP_MAX)  # noqa: S311
    time.sleep(initial_sleep)
    total_sleep_time += initial_sleep

    for i in range(total_steps):
        msg = ""
        step_sleep = random.uniform(STEP_SLEEP_MIN, STEP_SLEEP_MAX)  # noqa: S311
        time.sleep(step_sleep)
        total_sleep_time += step_sleep

        if i == total_steps // 2:
            msg = f"Task {task_id_num} is halfway done."

        progress_queue.put(
            ProgressMessage(
                type=ProgressMessageType.UPDATE,
                task_id=rich_task_id,
                advance=1,
                message=msg,
            ),
        )

    return total_sleep_time


def _manage_worker_progress[T](
    task_id_num: int,
    total_steps: int,
    shared_resources: SharedResources[T],
    task_logic_func: Callable[[int, int, TaskID | None, ManagerQueue[T]], T],
) -> float:
    available_task_ids: Manager.list = shared_resources.task_ids
    slot_lock: mp.Lock = shared_resources.lock
    progress_queue: mp.Queue = shared_resources.queue

    rich_task_id: TaskID | None = None
    acquired: bool = False
    task_result: float | None = None
    task_start_monotonic_time: float | None = None

    for _ in range(LOCK_ACQUISITION_RETRIES):
        if slot_lock.acquire(timeout=LOCK_ACQUISITION_TIMEOUT):
            try:
                if available_task_ids:
                    rich_task_id = available_task_ids.pop(0)
                    acquired = True
                break
            finally:
                slot_lock.release()
        else:
            time.sleep(LOCK_RETRY_SLEEP)

    if not acquired or rich_task_id is None:
        task_result = _perform_task_work(task_id_num, total_steps, None, progress_queue)
        progress_queue.put(
            ProgressMessage(
                type=ProgressMessageType.LOG,
                message=WARNING_NO_SLOT_MESSAGE.format(task_id_num, task_result),
            ),
        )
        return task_result

    try:
        task_start_monotonic_time = time.monotonic()
        progress_queue.put(
            ProgressMessage(
                type=ProgressMessageType.START,
                task_id=rich_task_id,
                description=f"Task {task_id_num}",
                total=total_steps,
                start_time=task_start_monotonic_time,
            ),
        )

        task_result = task_logic_func(
            task_id_num,
            total_steps,
            rich_task_id,
            progress_queue,
        )

    finally:
        progress_queue.put(
            ProgressMessage(
                type=ProgressMessageType.FINISH,
                task_id=rich_task_id,
                result_data=task_result,
            ),
        )

        with slot_lock:
            available_task_ids.append(rich_task_id)

    return task_result


# --- Worker Function (Entry point for multiprocessing pool) ---
def worker_function(task_info: tuple[int, int, SharedResources]) -> float:
    task_id_num, total_steps, shared_resources = task_info
    return _manage_worker_progress(
        task_id_num,
        total_steps,
        shared_resources,
        _perform_task_work,
    )


# --- Initialization and Setup Functions ---
def _initialize_shared_resources_and_tasks(
    manager: SyncManager,
    num_tasks: int,
    min_steps: int,
    max_steps: int,
) -> tuple[SharedResources, list[tuple[int, int, SharedResources]]]:
    shared_resources = SharedResources(
        task_ids=manager.list(),
        lock=manager.Lock(),
        queue=manager.Queue(),
    )

    task_args: list[tuple[int, int, SharedResources]] = []
    for i in range(num_tasks):
        task_args.append((i, random.randint(min_steps, max_steps), shared_resources))

    return shared_resources, task_args


def _setup_progress_display(
    console: Console,
    progress: Progress,
    num_processes: int,
    num_tasks: int,
    shared_resources: SharedResources,
) -> TaskID:
    pre_created_rich_tasks: list[TaskID] = []
    for i in range(num_processes):
        task_id = progress.add_task(
            DEFAULT_WAITING_TASK_DESCRIPTION,
            total=1,
            visible=False,
        )
        pre_created_rich_tasks.append(task_id)

    shared_resources.task_ids.extend(pre_created_rich_tasks)

    overall_task = progress.add_task(OVERALL_PROGRESS_DESCRIPTION, total=num_tasks)
    return overall_task


# --- Progress Update Processing Loop ---
def _process_progress_updates(
    progress: Progress,
    progress_queue: mp.Queue,
    overall_task: TaskID,
    num_tasks: int,
    console: Console,
    task_start_times: dict[TaskID, float],
) -> None:
    completed_tasks_count: int = 0
    while completed_tasks_count < num_tasks:
        try:
            update_message: ProgressMessage = progress_queue.get(
                timeout=QUEUE_GET_TIMEOUT,
            )

            if update_message.type == ProgressMessageType.START:
                if (
                    update_message.task_id is not None
                    and update_message.start_time is not None
                ):
                    task_start_times[update_message.task_id] = update_message.start_time
                progress.update(
                    update_message.task_id,
                    description=update_message.description,
                    total=update_message.total,
                    completed=0,
                    visible=True,
                )
            elif update_message.type == ProgressMessageType.UPDATE:
                if update_message.task_id is not None:
                    progress.update(
                        update_message.task_id,
                        advance=update_message.advance,
                    )
                    if update_message.message:
                        print(
                            f"Task {update_message.task_id}: {update_message.message}.",
                        )

            elif update_message.type == ProgressMessageType.FINISH:
                if update_message.task_id is not None:
                    progress.update(
                        update_message.task_id,
                        completed=progress.tasks[update_message.task_id].total,
                        visible=False,
                    )
                    if update_message.task_id in task_start_times:
                        del task_start_times[update_message.task_id]

                if update_message.result_data is not None:
                    print(
                        TASK_FINISHED_MESSAGE.format(update_message.result_data),
                    )

                progress.advance(overall_task)
                completed_tasks_count += 1
            elif update_message.type == ProgressMessageType.LOG:
                console.print(update_message.message)

        except mp.queues.Empty:
            pass
        except Exception as e:
            console.print(f"[red]Error processing progress update: {e}[/red]")


# --- Main Orchestration Function ---
def run_progress_manager(
    num_tasks: int,
    min_steps: int,
    max_steps: int,
    num_processes: int,
) -> list[float]:
    console = Console()
    console.print(
        f"Starting process pool with {num_processes} concurrent workers for {num_tasks} tasks...",
    )

    final_results: list[float] = []
    task_start_times: dict[TaskID, float] = {}

    with Manager() as manager:
        shared_resources, task_args = _initialize_shared_resources_and_tasks(
            manager,
            num_tasks,
            min_steps,
            max_steps,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            # Initialize TaskElapsedTimeColumn with a placeholder overall_task_id
            # This ID will be updated after the overall_task is created
            TaskElapsedTimeColumn(task_start_times, overall_task_id=TaskID(-1)),
            console=console,
            refresh_per_second=PROGRESS_REFRESH_RATE,
        ) as progress:
            overall_task = _setup_progress_display(
                console,
                progress,
                num_processes,
                num_tasks,
                shared_resources,
            )
            # Corrected: Assign overall_task (which is already a TaskID) directly
            progress.columns[-1].overall_task_id = overall_task

            with Pool(processes=num_processes) as pool:
                async_results = [
                    pool.apply_async(worker_function, (arg,)) for arg in task_args
                ]

                _process_progress_updates(
                    progress,
                    shared_resources.queue,
                    overall_task,
                    num_tasks,
                    console,
                    task_start_times,
                )

                final_results = [res.get() for res in async_results]

    console.print(ALL_PROCESSING_COMPLETE_MESSAGE)
    return final_results


def main() -> None:
    """Showcase the live progress manager with multiprocessing."""
    task_work_outputs = run_progress_manager(
        num_tasks=NUM_TASKS,
        min_steps=MIN_STEPS,
        max_steps=MAX_STEPS,
        num_processes=NUM_PROCESSES,
    )
    print(
        AVERAGE_SLEEP_TIME_FORMAT.format(
            sum(task_work_outputs) / len(task_work_outputs),
        ),
    )


if __name__ == "__main__":
    main()
