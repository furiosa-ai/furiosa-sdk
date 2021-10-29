from contextlib import contextmanager
import os
import sys
from typing import Iterator


def removeprefix(word: str, prefix: str) -> str:
    """Python 3.9 removeprefix().

    See https://docs.python.org/3/library/stdtypes.html#str.removeprefix
    """
    return word[len(prefix) :] if word.startswith(prefix) else word  # noqa: E203


@contextmanager
def python_path(directory: str) -> Iterator[None]:
    """Context adding the directory into PYTHONPATH."""
    sys.path.insert(0, directory)
    yield
    sys.path.remove(directory)


@contextmanager
def working_directory(directory: str) -> Iterator[None]:
    """Context replacing current working directory."""
    previous = os.getcwd()

    os.chdir(directory)
    yield
    os.chdir(previous)
