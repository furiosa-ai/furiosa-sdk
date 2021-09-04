import os
from pathlib import Path
import sys

from . import consts


def is_compile_log_enabled() -> bool:
    """
    Return True or False whether the compile log is enabled or not.

    Returns
        True if the compile log is enabled, or False.
    """
    return 'unittest' not in sys.modules.keys()


def furiosa_home() -> str:
    """
    Return FURIOSA_HOME which is the base directory of furiosa tools

    Returns:
        Furiosa home directory
    """
    return os.environ.get(consts.FURIOSA_HOME_ENV, f"{Path.home()}/furiosa")


def log_dir() -> str:
    """
    Return FURIOSA_HOME which is the base directory of furiosa tools

    Returns:
        The log directory of furiosa home
    """
    return os.environ.get(consts.FURIOSA_LOG_DIR_ENV, f"{furiosa_home()}/logs")


def current_npu_device() -> str:
    """
    Return the current npu device name

    Returns:
        NPU device name
    """
    return os.environ.get(consts.NPU_DEVNAME_ENV, consts.DEFAULT_DEVNAME)
