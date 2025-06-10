"""Debugging tools for Python.

This module provides functions to set up debugging tools like PuDB and Rich Traceback.
It checks for the availability of these tools and configures them accordingly.
"""


def use_pdub() -> None:
    """Check if PuDB is available and set it as the default debugger."""
    try:
        import pudb  # noqa: F401, T100
    except ImportError:
        pass
    else:
        # Set the default Python breakpoint to use PuDB for debugging
        # This allows you to use `breakpoint()` in your code to trigger the debugger
        import os

        if "PYTHONBREAKPOINT" not in os.environ:
            os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"


def use_rich_traceback() -> None:
    """Check if Rich Traceback is available and set it as the default."""
    try:
        from rich import traceback
    except ImportError:
        pass
    else:
        # Install rich traceback to enhance the debugging experience
        _ = traceback.install(show_locals=True)


def install_inspect() -> None:
    """Print the object using a custom inspect function."""
    try:
        from rich import inspect
    except ImportError:
        return
    else:
        import builtins

        builtins.i = inspect


def install() -> None:
    """Install debugging tools."""
    use_pdub()
    use_rich_traceback()
    install_inspect()


install()
