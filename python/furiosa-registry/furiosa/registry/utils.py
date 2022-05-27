from contextlib import contextmanager
import importlib
import logging
import os
import sys
import types
from typing import Iterator

Logger = logging.getLogger(__name__)


def removeprefix(word: str, prefix: str) -> str:
    """Python 3.9 removeprefix().

    See https://docs.python.org/3/library/stdtypes.html#str.removeprefix
    """
    return word[len(prefix) :] if word.startswith(prefix) else word  # noqa: E203


@contextmanager
def python_path(directory: str) -> Iterator[None]:
    """Context adding the directory into PYTHONPATH."""
    sys.path.insert(0, directory)
    try:
        yield
    finally:
        sys.path.remove(directory)


@contextmanager
def working_directory(directory: str) -> Iterator[None]:
    """Context replacing current working directory."""
    previous = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def import_module(directory: str, path: str) -> types.ModuleType:
    """Import module via specified path from the local directory."""

    # Replace working directory to use file system dependent function like 'open' in a registry.
    with python_path(directory), working_directory(directory):
        # Remove file extension .py: models/model.py -> models/model
        path = os.path.splitext(path)[0]
        # Replace slash(/) with dot(.): models/model -> models.model
        path = path.replace("/", ".")

        if path in sys.modules:
            # Remove if module with same name already exists
            sys.modules.pop(path)

        try:
            return importlib.import_module(path)
        except ModuleNotFoundError as e:
            Logger.error(
                f"Module dependencies for the loaded code not found. "
                f"You should install required dependencies for loaded module '{path}'. "
                f"See error trace to identify the missing module"
            )
            raise e
