"""Tree walker for Pydantic models.

Creates printout of a Pydantic model structure as a tree.
"""

from collections.abc import Generator
from dataclasses import dataclass, fields
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from rich.console import Console

console = Console()
colors = ["navy_blue", "blue", "cyan", "green", "yellow", "orange", "red"]
type_colour = "bright_black"


def is_pydantic_model(type_: type[Any]) -> bool:
    """Check whether a type is a Pydantic model."""
    return hasattr(type_, "model_fields")


def is_dataclass(type_: type[Any]) -> bool:
    """Check whether a type is a dataclass."""
    return hasattr(type_, "__dataclass_fields__")


def is_model(type_: type[Any]) -> bool:
    """Check whether a type is a Pydantic model or a dataclass."""
    return is_pydantic_model(type_) or is_dataclass(type_)


def model_fields(type_: type[Any]) -> Generator[tuple[str, type[Any]]]:
    """Yield the fields and their types of a given type."""
    if is_pydantic_model(type_):
        for name, info in type_.model_fields.items():
            yield name, info.annotation
    if is_dataclass(type_):
        for field in fields(type_):
            yield field.name, field.type


def type_to_string(type_: type[Any]) -> str:
    """Convert a type to a string representation."""
    orgin = get_origin(type_)
    if orgin is None:
        return type_.__name__
    args_str = ",".join(type_to_string(t) for t in get_args(type_))
    return f"{type_to_string(orgin)}[{args_str}]"


def extract_model_types_from_composite(
    type_: type[Any],
) -> Generator[type[Any]]:
    """Recursively extracts Pydantic model types from a composite type."""
    for arg in get_args(type_):
        if is_model(arg):
            yield arg
        else:
            yield from extract_model_types_from_composite(arg)


def tree(
    type_: type,
    level: int = 0,
    *,
    is_root: bool = True,
) -> Generator[tuple[str, str, int]]:
    """Generate a tree representation of a model's fields and their types.

    Recursively traverses the fields of a Pydantic model or dataclass,
    yielding the field name, its type as a string, and the level of depth
    in the tree. This allows for a structured view of nested models.

    Args:
        type_ (type): The model type to generate the tree for.
        level (int, optional): The current depth level in the tree.
            Defaults to 0.
        is_root (bool, optional): Flag indicating if the current node is the root.
            Defaults to True.

    Yields:
        tuple[str, str, int]: A tuple containing the field name, its type as a string,
            and the level of depth.

    """
    if is_root:
        # If root there is no field name
        yield type_to_string(type_), "", level
        yield from tree(type_, level, is_root=False)
    else:
        for name, field_type_ in model_fields(type_):
            yield name, type_to_string(field_type_), level + 1
            # Yield from fields of the current type, if it is a model
            if is_model(field_type_):
                yield from tree(field_type_, level + 1, is_root=False)
            # Check if the field is a composite type, and if so, yield from its models
            for factor_type_ in extract_model_types_from_composite(field_type_):
                # Args of a composite type are also treated as roots
                yield from tree(factor_type_, level + 2, is_root=True)


def console_print(indent: str, name: str, type_: str) -> None:
    """Print a single line of a tree representation of a model to the console."""
    level_color = len(indent) % len(colors)
    console.print(
        indent,
        f"[{colors[level_color]}]{name.ljust(7)}[/{colors[level_color]}]",
        f"[{type_colour}]{type_}[/{type_colour}]",
    )


def display(model: type, *, rich: bool = False) -> None:
    """Print a tree representation of a model to the console.

    The tree is represented as a series of lines, where each line
    represents a field in the model. The indentation of each line
    corresponds to its level in the tree, and the text of the line
    contains the field name and its type.
    """
    for name, type_, level in tree(model):
        indent = "   " * level
        name_padded = name.ljust(8) + ": "
        worker = console_print if rich else print
        worker(indent, name_padded, type_)


if __name__ == "__main__":

    class InnerNode(BaseModel):
        """Inner node model for testing."""

        name: str
        id_number: int

    class AnotherInnerNode(BaseModel):
        """Another inner node model for testing."""

        age: int
        weight: float

    @dataclass
    class InnerDataclass:
        """Inner dataclass for testing."""

        name: str
        id_number: int

    @dataclass
    class NodeDataclass:
        """Node dataclass for testing."""

        name: str
        inner: InnerDataclass

    class Node(BaseModel):
        """Node model for testing."""

        name: str
        inner: InnerNode
        many_inner: list[InnerNode]
        union_inner: InnerNode | AnotherInnerNode
        many_union_inner: list[InnerNode | AnotherInnerNode]
        dict_union_inner: dict[str, InnerNode]

    class Root(BaseModel):
        """Root model for testing."""

        name: dict[str, InnerNode]
        data_child: NodeDataclass
        child: Node

    display(Root, rich=True)
