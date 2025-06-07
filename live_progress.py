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
from enum import StrEnum, auto
from functools import partial
from multiprocessing import Manager, Pool
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Queue as _ManagerQueue
from multiprocessing.synchronize import Lock as ManagerLock
from typing import cast, override

from rich.align import Align
from rich.console import Console, ConsoleOptions
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
STEP_SLEEP_MAX: float = 0.05

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
type Worker[T] = Callable[[int, int, TaskID | None, ManagerQueue[T]], T]


@dataclass
class SharedResources[T]:
    """Dataclass to hold shared resources for multiprocessing tasks."""

    task_ids: list[int]
    lock: ManagerLock
    queue: ManagerQueue[T]


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


from rich.console import RenderableType


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
        if not self.messages:
            inner_content = Align.center(Text("No messages yet...", style="dim italic"))
        else:
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
    shared_resources: SharedResources[T],
    task_logic_func: Worker[T],
) -> T:
    """'Manage the progress of a worker task, acquiring a slot and reporting progress."""
    available_task_ids = shared_resources.task_ids
    slot_lock = shared_resources.lock
    progress_queue = shared_resources.queue

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


def _managed_worker[T](
    task_info: tuple[int, int, SharedResources[T]],
    worker: Worker[T],
) -> T:
    task_id_num, total_steps, shared_resources = task_info
    return _manage_worker_progress(
        task_id_num,
        total_steps,
        shared_resources,
        worker,
    )


def _initialize_shared_resources_and_tasks[T](
    manager: SyncManager,
    num_tasks: int,
    min_steps: int,
    max_steps: int,
) -> tuple[SharedResources[T], list[tuple[int, int, SharedResources[T]]]]:
    shared_resources: SharedResources[T] = SharedResources(
        task_ids=manager.list(),  # pyright: ignore[reportArgumentType]
        lock=manager.Lock(),  # pyright: ignore[reportArgumentType]
        queue=cast("ManagerQueue[T]", manager.Queue()),  # pyright: ignore[reportInvalidCast]
    )

    task_args = [
        (i, random.randint(min_steps, max_steps), shared_resources)  # noqa: S311
        for i in range(num_tasks)
    ]
    return shared_resources, task_args


def _setup_progress_display[T](
    console: Console,
    progress: Progress,
    num_processes: int,
    num_tasks: int,
    shared_resources: SharedResources[T],
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


# --- Progress Update Processing Loop ---
@dataclass
class ProgressUpdateContext:
    """Context object for managing progress updates in the progress manager."""

    progress: Progress
    message_panel: MessagePanel
    overall_task: TaskID
    num_tasks: int
    console: Console
    task_start_times: dict[TaskID, float]


def _handle_start_message[T](
    msg: ProgressMessage[T],
    ctx: ProgressUpdateContext,
) -> None:
    if msg.task_id is not None and msg.start_time is not None:
        ctx.task_start_times[msg.task_id] = msg.start_time
    ctx.progress.update(
        msg.task_id,
        description=msg.description,
        total=msg.total,
        completed=0,
        visible=True,
    )


def _handle_update_message[T](
    msg: ProgressMessage[T],
    ctx: ProgressUpdateContext,
) -> None:
    if msg.task_id is not None:
        ctx.progress.update(
            msg.task_id,
            advance=msg.advance,
        )
        if msg.message:
            ctx.message_panel.add_message(
                f"Task {msg.task_id}: {msg.message}.",
            )


def _handle_finish_message[T](
    msg: ProgressMessage[T],
    ctx: ProgressUpdateContext,
) -> int:
    if msg.task_id is not None:
        ctx.progress.update(
            msg.task_id,
            completed=ctx.progress.tasks[msg.task_id].total,
            visible=False,
        )
        if msg.task_id in ctx.task_start_times:
            del ctx.task_start_times[msg.task_id]

    if msg.result_data is not None:
        ctx.message_panel.add_message(
            TASK_FINISHED_MESSAGE.format(msg.result_data),
        )

    ctx.progress.advance(ctx.overall_task)
    return 1


def _handle_log_message[T](
    msg: ProgressMessage[T],
    ctx: ProgressUpdateContext,
) -> None:
    if msg.message is not None:
        ctx.message_panel.add_message(msg.message)


def _process_progress_updates[T](
    progress: Progress,
    message_panel: MessagePanel,
    progress_queue: ManagerQueue[T],
    overall_task: TaskID,
    num_tasks: int,
    console: Console,
    task_start_times: dict[TaskID, float],
) -> None:
    ctx = ProgressUpdateContext(
        progress=progress,
        message_panel=message_panel,
        overall_task=overall_task,
        num_tasks=num_tasks,
        console=console,
        task_start_times=task_start_times,
    )
    completed_tasks_count: int = 0
    handlers = {
        ProgressMessageType.START: _handle_start_message,
        ProgressMessageType.UPDATE: _handle_update_message,
        ProgressMessageType.FINISH: _handle_finish_message,
        ProgressMessageType.LOG: _handle_log_message,
    }
    while completed_tasks_count < num_tasks:
        try:
            update_message: ProgressMessage[T] = progress_queue.get(
                timeout=QUEUE_GET_TIMEOUT,
            )
            handler = handlers.get(update_message.type)
            if handler is not None:
                if update_message.type == ProgressMessageType.FINISH:
                    completed_tasks_count += handler(update_message, ctx)
                else:
                    handler(update_message, ctx)
        except mp.queues.Empty:
            pass
        except Exception as e:
            message_panel.add_message(
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
        shared_resources, task_args = _initialize_shared_resources_and_tasks(
            manager,
            num_tasks,
            min_steps,
            max_steps,
        )
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
            console,
            progress,
            num_processes,
            num_tasks,
            shared_resources,
        )
        message_panel = MessagePanel(max_messages=20)
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
            message_panel,
        )
        with Live(progress_table, screen=True, refresh_per_second=10, transient=False):
            # Corrected: Assign overall_task (which is already a TaskID) directly
            progress.columns[-1].overall_task_id = overall_task

            with Pool(processes=num_processes) as pool:
                message_panel.add_message(
                    f"Starting process pool with {num_processes} "
                    f"workers for {num_tasks} tasks.",
                )

                async_results = [
                    pool.apply_async(partial(_managed_worker, worker=worker), (arg,))
                    for arg in task_args
                ]

                _process_progress_updates(
                    progress,
                    message_panel,
                    shared_resources.queue,
                    overall_task,
                    num_tasks,
                    console,
                    task_start_times,
                )

                final_results = [res.get() for res in async_results]

                message_panel.add_message(ALL_PROCESSING_COMPLETE_MESSAGE)
            _ = input()
    return final_results


def fake_work(
    task_id_num: int,
    total_steps: int,
    rich_task_id: TaskID | None,
    progress_queue: ManagerQueue[float],
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
