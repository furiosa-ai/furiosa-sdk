from __future__ import annotations

import os
from pathlib import Path
import warnings

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


def profiler_output() -> None | str:
    """
    Return FURIOSA_PROFILER_OUTPUT_PATH where profiler outputs written.

    For compatibility, NUX_PROFILER_PATH is also currently being supported, but it will be
    deprecated by FURIOSA_PROFILER_OUTPUT_PATH later.

    Returns:
        The file path of profiler output if specified, or None.
    """
    if os.environ.get("NUX_PROFILER_PATH") is not None:
        warnings.warn("NUX_PROFILER_PATH is deprecated, use FURIOSA_PROFILER_OUTPUT_PATH instead")

    return os.environ.get(
        consts.ENV_FURIOSA_PROFILER_OUTPUT_PATH, os.environ.get("NUX_PROFILER_PATH")
    )
