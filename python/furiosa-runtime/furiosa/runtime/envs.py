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
    return not os.environ.get('FURIOSA_TEST_IS_RUNNING')


def xdg_state_home() -> str:
    """
    Return XDG_STATE_HOME which is the base directory of furiosa logs, history, and other states

    Returns:
        Furiosa home directory
    """
    return os.environ.get(consts.ENV_XDG_STATE_HOME, f"{Path.home()}/.local/state/furiosa")


def log_dir() -> str:
    """
    Return FURIOSA_LOG_DIR where the logs are stored.

    Returns:
        The log directory of furiosa sdk
    """
    return os.environ.get(consts.ENV_FURIOSA_LOG_DIR, f"{xdg_state_home()}/logs")


def current_npu_device() -> str:
    """
    Return the current npu device name

    Returns:
        NPU device name
    """
    return os.environ.get(consts.ENV_NPU_DEVNAME, consts.DEFAULT_DEVNAME)
