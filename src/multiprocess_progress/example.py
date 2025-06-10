"""Example of using the live progress manager with multiprocessing."""

from rich.progress import TaskID

from src.multiprocess_progress.live_progress import ManagerQueue, run_progress_manager

# --- Constants -------
# Task Simulation Constants
INITIAL_SLEEP_MIN: float = 0.1
INITIAL_SLEEP_MAX: float = 0.5
STEP_SLEEP_MIN: float = 0.01
STEP_SLEEP_MAX: float = 0.02
NUM_TASKS: int = 20
MIN_STEPS: int = 50
MAX_STEPS: int = 200
NUM_PROCESSES: int = 10


def fake_work(
    task_id_num: int,
    total_steps: int,
    rich_task_id: TaskID,
    progress_queue: ManagerQueue,
) -> float:
    total_sleep_time = 0.0

    initial_sleep = random.uniform(INITIAL_SLEEP_MIN, INITIAL_SLEEP_MAX)
    time.sleep(initial_sleep)
    total_sleep_time += initial_sleep

    for i in range(total_steps):
        msg = ""
        step_sleep = random.uniform(STEP_SLEEP_MIN, STEP_SLEEP_MAX)
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


def fake_work(
    task_id_num: int,
    total_steps: int,
    rich_task_id: TaskID,
    progress_queue: ManagerQueue,
) -> float:
    total_sleep_time = 0.0

    initial_sleep = random.uniform(INITIAL_SLEEP_MIN, INITIAL_SLEEP_MAX)
    time.sleep(initial_sleep)
    total_sleep_time += initial_sleep

    for i in range(total_steps):
        msg = ""
        step_sleep = random.uniform(STEP_SLEEP_MIN, STEP_SLEEP_MAX)
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
