from ctypes import CDLL, c_char_p, c_int, util
from enum import IntEnum
import glob
import logging
import os
from sys import platform
from typing import Callable, Optional

LOG = logging.getLogger(__name__)


DEFAULT_ENCODING: str = "utf-8"


def find_user_lib_path(libname: str):
    libpath = None
    for lib_path in os.getenv('LD_LIBRARY_PATH').split(":"):
        if platform == "linux":
            for name in glob.glob(f"{lib_path}/lib{libname}.so*"):
                libpath = name
                break
        elif platform == "darwin":
            for name in glob.glob(f"{lib_path}/lib{libname}.dylib*"):
                libpath = name
                break

    return libpath


def find_global_lib_path(libname: str):
    return util.find_library(libname)


def find_native_lib_path(libname: str) -> Optional[str]:
    """Finding a native library path according to the following priority
    1. If the environment variable 'LD_LIBRARY_PATH' is set,
    this function tries to find native library found from LD_LIBRARY_PATH.
    2. Otherwise, it tries find the native library from global library paths,
    such as /usr/lib, /usr/local/lib.
    3. If the library still cannot be founded, it returns None.
    """
    libpath = None

    if os.getenv('LD_LIBRARY_PATH'):
        libpath = find_user_lib_path(libname)

    if not libpath:
        libpath = find_global_lib_path(libname)

    return libpath


def __register_common_capis(ciface):
    ciface.version.argtypes = []
    ciface.version.restype = c_char_p

    ciface.git_short_hash.argtypes = []
    ciface.git_short_hash.restype = c_char_p

    ciface.build_timestamp.argtypes = []
    ciface.build_timestamp.restype = c_char_p

    ciface.register_signal_handler.argtypes = []
    ciface.register_signal_handler.restype = None

    ciface.enable_logging.argtypes = [c_int]
    ciface.enable_logging.restype = None


def find_native_libs(libname: str, register_hook: Callable = __register_common_capis):
    """Finding a native library c interface according to the following priority
    1. If the environment variable 'LD_LIBRARY_PATH' is set,
    this function tries to find native library found from LD_LIBRARY_PATH.
    2. Otherwise, it tries to find the native library embedded in the python package.
    3. If the embedded native library cannot be found,
    it tries find the native library from global library paths, such as /usr/lib, /usr/local/lib.

    After loading C Library, it will call register_hook(ciface).
    """

    libpath = find_native_lib_path(libname)
    if not libpath:
        raise SystemExit(f'fail to find lib{libname}')

    ciface = CDLL(libpath)

    if ciface:
        register_hook(ciface)
        LOG.info(
            'loaded native library {} ({} {})'.format(
                libpath,
                ciface.version().decode(DEFAULT_ENCODING),
                ciface.git_short_hash().decode(DEFAULT_ENCODING),
            )
        )
    else:
        raise SystemExit('fail to load native library')

    return ciface


class LogLevel(IntEnum):
    """Python object correspondnig to nux_log_level_t in Nux C API"""

    OFF = 0
    ERROR = 1
    WARN = 2
    INFO = 3
