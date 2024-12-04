"""Utilities for working with iterables.

Public functions:
    batched
"""

from collections.abc import Iterable
from itertools import islice
from sys import maxsize
from typing import TypeVar

T = TypeVar("T")
S = TypeVar("S")


def batched(
    iterable: Iterable[T],
    n: int,
    *,
    strict: bool = False,
) -> Iterable[tuple[T, ...]]:
    """Split an iterable into batches of size n.

    Args:
        iterable: The iterable to split into batches.
        n: The size of each batch.
        strict: If True, raise a ValueError if the last batch is incomplete.

    Returns:
        An iterator over the batches.

    >>>
    >>> for batch in batched("ABCDEFG", 3):
    ...     print(batch)
    ('A', 'B', 'C')s
    ('D', 'E', 'F')
    ('G',)

    """
    if not 1 <= n <= maxsize:
        msg = "Batch size n must be at least one and at most sys.maxsize."
        raise ValueError(msg)

    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            msg = "Incomplete batch for strict batching."
            raise ValueError(msg)
        yield batch


def unbox(iterable: Iterable[T]) -> T:
    """Unbox an iterable if it contains a single element.

    Args:
        iterable: The iterable to unbox.

    Returns:
        The single element of the iterable.

    >>> unbox([1]) == 1
    True
    >>> unbox([1, 2])
    Traceback (most recent call last):
        ...
    ValueError: Iterable contains more than one element.
    >>> unbox((1,)) == 1
    True
    >>> unbox((i for i in range(1))) == 0
    True

    """
    if not isinstance(iterable, Iterable):
        msg = "Input must be an iterable."
        raise TypeError(msg)

    iterator = iter(iterable)
    first_value = next(iterator)
    try:
        _ = next(iterator)
        msg = "Iterable contains more than one element."
        raise ValueError(msg)
    except StopIteration:
        return first_value


def unzip(iterable: Iterable[tuple[T, S]]) -> tuple[Iterable[T], Iterable[S]]:
    """Unzip an iterable of tuples.

    Args:
        iterable: The iterable to unzip.

    Returns:
        A tuple of iterables.

    >>> unzip([(1, "a"), (2, "b"), (3, "c")]) == ([1, 2, 3], ["a", "b", "c"])
    True

    """
    first, second = zip(*iterable, strict=True)
    return first, second
