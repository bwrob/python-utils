"""Utilities for working with iterables.

Public functions:
    batched
"""

from collections.abc import Iterable
from itertools import islice
from sys import maxsize
from typing import Any, TypeVar

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


def unzip(iterable: Iterable[tuple[Any, ...]]) -> tuple[Iterable[Any], ...]:
    """Unzip an iterable of tuples.

    Args:
        iterable: The iterable to unzip.

    Returns:
        A tuple of iterables.

    >>> unzip([(1, "a"), (2, "b"), (3, "c")]) == ([1, 2, 3], ["a", "b", "c"])
    True

    """
    zipped = zip(*iterable, strict=True)
    return tuple(zipped)


def flatten_iterable(iterable: Iterable[Any]) -> list[Any]:
    """Flatten an iterable of iterables.

    >>> flatten_iterable([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    [1, 2, 3, 4, 5, 6, 7, 8]
    >>> flatten_iterable([[1, [2, 3]], 4, 5])
    [1, 2, 3, 4, 5]

    """
    return [
        item
        for sublist in iterable
        for item in (
            flatten_iterable(sublist)
            if isinstance(sublist, Iterable) and not isinstance(sublist, str)
            else [sublist]
        )
    ]


def flatten_string_key_dict(dictionary: dict[str, Any]) -> dict[str, Any]:
    """Flatten a dictionary of strings and nested dictionaries of strings and objects.

    >>> flatten_string_key_dict({"a": "b", "c": {"d": "e"}}) == {"a": "b", "c.d": "e"}
    True

    """
    return {
        key + "." + subkey if isinstance(subvalue, dict) else key: subvalue
        for key, value in dictionary.items()
        for subkey, subvalue in flatten_string_key_dict(value).items()
    }
