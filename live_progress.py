"""Live progress manager using multiprocessing and rich for visual feedback.

This module demonstrates how to manage and display live progress bars for multiple
concurrent tasks using Python's multiprocessing and the rich library.
"""

import multiprocessing as mp
import random
import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import partial
from multiprocessing import Manager, Pool
from multiprocessing.queues import Queue as _ManagerQueue
from multiprocessing.synchronize import Lock as ManagerLock
from typing import cast, override

from rich.align import Align
from rich.console import Console, ConsoleOptions, RenderableType
from rich.live import Live
from rich.panel import Panel
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
from rich.table import Table
from rich.text import Text

# --- Constants -------
# Task Simulation Constants
INITIAL_SLEEP_MIN: float = 0.1
INITIAL_SLEEP_MAX: float = 0.5
STEP_SLEEP_MIN: float = 0.01
STEP_SLEEP_MAX: float = 0.02

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


NUM_TASKS: int = 20
MIN_STEPS: int = 50
MAX_STEPS: int = 200
NUM_PROCESSES: int = 10


@dataclass
class LogMessage:
    """Dataclass to represent a message sent through the progress queue."""

    task_id: TaskID
    message: str


@dataclass
class StartMessage:
    """Dataclass to represent a message sent through the progress queue."""

    task_id: TaskID
    description: str
    start_time: float
    total: int


@dataclass
class UpdateMessage:
    """Dataclass to represent a message sent through the progress queue."""

    task_id: TaskID
    message: str
    advance: int


@dataclass
class FinishMessage:
    """Dataclass to represent a message sent through the progress queue."""

    task_id: TaskID


type ProcessMessage = LogMessage | StartMessage | UpdateMessage | FinishMessage
type ManagerQueue = _ManagerQueue[ProcessMessage]
type Worker[T] = Callable[[int, int, TaskID, ManagerQueue], T]


@dataclass
class SharedResources:
    """Dataclass to hold shared resources for multiprocessing tasks."""

    task_ids: list[TaskID]
    lock: ManagerLock
    queue: ManagerQueue


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
            elapsed_time = task.elapsed
        elif task.id in self.task_start_times:
            elapsed_time = time.monotonic() - self.task_start_times[task.id]
        else:
            return Text("-:--:--")
        return Text(str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))


class MessagePanel(Panel):
    """A panel to display a live log of messages."""

    def __init__(
        self,
        max_messages: int = 10,
        title: str = "[bold green]Live Message Log[/bold green]",
        border_style: str = "green",
    ) -> None:
        """Initialize the MessagePanel.

        Args:
            max_messages: Maximum number of messages to display.
            title: Title of the panel.
            border_style: Border style for the panel.
            **kwargs: Additional keyword arguments for Panel.

        """
        self.messages: deque[str] = deque(maxlen=max_messages)
        self.max_messages: int = max_messages
        self.console: Console = Console()

        super().__init__(
            Align.center(Text("No messages yet...", style="dim italic")),
            title=title,
            border_style=border_style,
            expand=True,
            subtitle="[dim]Displaying last 0 messages[/dim]",
            height=self.max_messages + 2,  # +2 for title and subtitle
        )

    def add_message(self, message: str) -> None:
        """Add a message to the panel.

        Args:
            message: The message string to add.

        """
        self.messages.append(message)

    @override
    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> Generator[RenderableType]:
        """Render the panel with the latest messages.

        Args:
            console: The console instance.
            options: Console rendering options.

        Yields:
            RenderableType: The rendered panel.

        """
        message_lines: list[Text] = []
        for i, msg in enumerate(self.messages):
            if i == len(self.messages) - 1:
                message_lines.append(Text(f"• {msg}", style="bold yellow"))
            elif i == len(self.messages) - 2:
                message_lines.append(Text(f"• {msg}", style="orange3"))
            else:
                message_lines.append(Text(f"• {msg}", style="grey50"))
        inner_content = Text("\n").join(message_lines)

        self.renderable: RenderableType = inner_content
        self.subtitle: Text | str | None = Text(
            f"Displaying last {len(self.messages)} messages",
            style="dim",
        )

        yield from cast(
            "Generator[RenderableType]",
            super().__rich_console__(console, options),
        )


def _manage_worker_progress[T](
    task_id_num: int,
    total_steps: int,
    shared_resources: SharedResources,
    task_logic_func: Worker[T],
) -> T:
    """'Manage the progress of a worker task, acquiring a slot and reporting progress."""
    available_task_ids = shared_resources.task_ids
    slot_lock = shared_resources.lock
    progress_queue = shared_resources.queue

    rich_task_id: TaskID | None = None
    acquired: bool = False
    task_result: T | None = None
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
        return task_logic_func(task_id_num, total_steps, None, progress_queue)

    try:
        task_start_monotonic_time = time.monotonic()
        progress_queue.put(
            StartMessage(
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
            FinishMessage(
                task_id=rich_task_id,
            ),
        )

        with slot_lock:
            available_task_ids.append(rich_task_id)

    return task_result


def _managed_worker[T](
    task_info: tuple[int, int, SharedResources],
    worker: Worker[T],
) -> T:
    task_id_num, total_steps, shared_resources = task_info
    return _manage_worker_progress(
        task_id_num,
        total_steps,
        shared_resources,
        worker,
    )


def _setup_progress_display(
    progress: Progress,
    num_processes: int,
    num_tasks: int,
    shared_resources: SharedResources,
) -> TaskID:
    pre_created_rich_tasks: list[TaskID] = []
    for _ in range(num_processes):
        task_id = progress.add_task(
            DEFAULT_WAITING_TASK_DESCRIPTION,
            total=1,
            visible=False,
        )
        pre_created_rich_tasks.append(task_id)

    shared_resources.task_ids.extend(pre_created_rich_tasks)

    overall_task = progress.add_task(OVERALL_PROGRESS_DESCRIPTION, total=num_tasks)
    return overall_task


def _handle_start_msg(
    msg: StartMessage,
    progress: Progress,
    log_panel: MessagePanel,
    task_start_times: dict[TaskID, float],
) -> None:
    task_start_times[msg.task_id] = msg.start_time
    progress.update(
        msg.task_id,
        description=msg.description,
        total=msg.total,
        completed=0,
        visible=True,
    )
    log_panel.add_message(f"Task {msg.task_id} - {msg.description} started.")


def _handle_update_msg(
    msg: UpdateMessage,
    progress: Progress,
    log_panel: MessagePanel,
) -> None:
    progress.update(
        msg.task_id,
        advance=msg.advance,
    )
    if msg.message:
        log_panel.add_message(
            f"Task {msg.task_id}: {msg.message}.",
        )


def _handle_finish_msg(
    msg: FinishMessage,
    progress: Progress,
    log_panel: MessagePanel,
    overall_task: TaskID,
    task_start_times: dict[TaskID, float],
) -> None:
    progress.update(
        msg.task_id,
        completed=progress.tasks[msg.task_id].total,
        visible=False,
    )
    if msg.task_id in task_start_times:
        del task_start_times[msg.task_id]

    total_time = time.monotonic() - task_start_times.get(msg.task_id, 0.0)
    log_panel.add_message(
        f"Task {msg.task_id} finished. Total sleep: {total_time:.2f}s",
    )
    progress.advance(overall_task)


def _handle_log_msg(
    msg: LogMessage,
    log_panel: MessagePanel,
) -> None:
    log_panel.add_message(
        f"Task {msg.task_id}: {msg.message}.",
    )


def _process_progress_updates(
    progress: Progress,
    log_panel: MessagePanel,
    progress_queue: ManagerQueue,
    overall_task: TaskID,
    num_tasks: int,
    task_start_times: dict[TaskID, float],
) -> None:
    completed_tasks_count: int = 0
    while completed_tasks_count < num_tasks:
        try:
            update_message: ProcessMessage = progress_queue.get(
                timeout=QUEUE_GET_TIMEOUT,
            )
            if isinstance(update_message, LogMessage):
                _handle_log_msg(update_message, log_panel)
            elif isinstance(update_message, StartMessage):
                _handle_start_msg(update_message, progress, log_panel, task_start_times)
            elif isinstance(update_message, UpdateMessage):
                _handle_update_msg(update_message, progress, log_panel)
            else:
                _handle_finish_msg(
                    update_message,
                    progress,
                    log_panel,
                    overall_task,
                    task_start_times,
                )
                completed_tasks_count += 1
        except mp.queues.Empty:
            pass
        except Exception as e:
            log_panel.add_message(
                f"[red]Error processing progress update: {e}[/red]",
            )


def run_progress_manager[T](
    *,
    num_tasks: int,
    min_steps: int,
    max_steps: int,
    num_processes: int,
    worker: Worker[T],
) -> list[T]:
    console = Console()

    task_start_times: dict[TaskID, float] = {}

    with Manager() as manager:
        shared_resources: SharedResources = SharedResources(
            task_ids=manager.list(),  # pyright: ignore[reportArgumentType]
            lock=manager.Lock(),  # pyright: ignore[reportArgumentType]
            queue=cast("ManagerQueue", manager.Queue()),  # pyright: ignore[reportInvalidCast]
        )

        task_args = [
            (i, random.randint(min_steps, max_steps), shared_resources)  # noqa: S311
            for i in range(num_tasks)
        ]
        print((i, j) for i, j, _ in task_args)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TaskElapsedTimeColumn(task_start_times, overall_task_id=TaskID(-1)),
            console=console,
            refresh_per_second=PROGRESS_REFRESH_RATE,
        )
        overall_task = _setup_progress_display(
            progress,
            num_processes,
            num_tasks,
            shared_resources,
        )
        progress.columns[-1].overall_task_id = overall_task
        log_panel = MessagePanel(max_messages=20)
        progress_table = Table.grid()
        progress_table.add_row(
            Panel(
                progress,
                title="Overall Progress",
                border_style="green",
                padding=(2, 2),
                expand=True,
                height=num_processes + 10,
            ),
        )
        progress_table.add_row(
            log_panel,
        )
        with Live(progress_table, screen=True, refresh_per_second=10, transient=False):
            with Pool(processes=num_processes) as pool:
                log_panel.add_message(
                    f"Starting pool with {num_processes} "
                    f"workers for {num_tasks} tasks.",
                )

                async_results = [
                    pool.apply_async(partial(_managed_worker, worker=worker), (arg,))
                    for arg in task_args
                ]

                _process_progress_updates(
                    progress=progress,
                    log_panel=log_panel,
                    progress_queue=shared_resources.queue,
                    overall_task=overall_task,
                    num_tasks=num_tasks,
                    task_start_times=task_start_times,
                )

                final_results = [res.get() for res in async_results]

            log_panel.add_message("All tasks have been completed.")
            progress.stop_task(overall_task)
            _ = input()
    return final_results


def fake_work(
    task_id_num: int,
    total_steps: int,
    rich_task_id: TaskID,
    progress_queue: ManagerQueue,
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
            UpdateMessage(
                task_id=rich_task_id,
                advance=1,
                message=msg,
            ),
        )

    return total_sleep_time


def main() -> None:
    """Showcase the live progress manager with multiprocessing."""
    task_work_outputs = run_progress_manager(
        num_tasks=NUM_TASKS,
        min_steps=MIN_STEPS,
        max_steps=MAX_STEPS,
        num_processes=NUM_PROCESSES,
        worker=fake_work,
    )
    print(
        AVERAGE_SLEEP_TIME_FORMAT.format(
            sum(task_work_outputs) / len(task_work_outputs),
        ),
    )


if __name__ == "__main__":
    main()
