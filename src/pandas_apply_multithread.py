"""Example how to use pandas apply with multithreading."""

import multiprocessing as mp
from logging import getLogger

import numpy as np
import pandas as pd

logger = getLogger(__name__)

SIZE = 1_000_000


def example_function(a: float, b: float, c: float, d: float) -> float:
    """Calculate an example return."""
    return a * b + c * d


def row_function(row: pd.Series) -> float:
    """Apply example function to row."""
    return example_function(row.A, row.B, row.C, row.D)


def wrapp_apply(df: pd.DataFrame) -> pd.Series:
    """Wrappes application of example function."""
    return df.apply(row_function, axis=1)


def apply_multi(data_frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Apply example function with multithreading."""
    num_of_processes = mp.cpu_count()
    data_split = np.array_split(data_frame, num_of_processes)
    pool = mp.Pool(num_of_processes)
    pool_results = pool.map(wrapp_apply, data_split)
    pool.close()
    pool.join()

    results: pd.Series = pd.concat(pool_results, axis=0)
    data_frame_with_results = pd.concat([data_frame, results], axis=1)
    data_frame_with_results.columns = list(data_frame_with_results.columns.values)[
        :-1
    ] + [column_name]
    return data_frame_with_results


def example_dataframe(size: int) -> pd.DataFrame:
    """Create example dataframe."""
    random_generator = np.random.default_rng(42)
    return pd.DataFrame(
        random_generator.integers(0, 100, size=(size, 4)),
        columns=list("ABCD"),
    )


def apply_mp(
    frame: pd.DataFrame,
) -> None:
    """Run main function."""
    _ = apply_multi(frame, "OUTPUT")


def apply(
    frame: pd.DataFrame,
) -> None:
    """Run main function."""
    _ = frame.apply(row_function, axis=1)


if __name__ == "__main__":
    frame = example_dataframe(size=SIZE)
    apply_mp(frame)
    apply(frame)
