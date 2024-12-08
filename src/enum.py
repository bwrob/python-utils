"""Nicer Enum class."""

from enum import StrEnum
from typing import override


class MyEnum(StrEnum):
    """My Enum class.

    * custom __missing__ method that ignores case
    * custom _generate_next_value_ method that returns uppercase self for use of auto()

    >>> class TestEnum(MyEnum):
    ...     test = auto()
    ...     enumz = auto()

    >>> TestEnum.list()
    ['test', 'enumz']

    >>> TestEnum("Test")
    TestEnum.test

    """

    @override
    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[str],
    ) -> str:
        """Return the upper-cased version of the member name.

        Used by auto().
        """
        return name.upper()

    @classmethod
    def list(cls) -> list[str]:
        """Return a list of all enum member names."""
        return [member.name for member in cls]

    @override
    @classmethod
    def _missing_(cls, value: object) -> StrEnum | None:
        if not isinstance(value, str):
            return None

        value = value.upper()
        for member in cls:
            if member.name.lower() == value:
                return member

        return None
