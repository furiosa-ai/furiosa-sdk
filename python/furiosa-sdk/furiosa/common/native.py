import glob
import logging
import os
import sys
from ctypes import CDLL, c_char_p, util, c_int
from enum import IntEnum
from sys import platform


LOG = logging.getLogger(__name__)


def _find_local_lib_path():
    module_path = os.path.dirname(__file__)
    if platform == "linux":
        for name in glob.glob("{}/runtime.cpython-*.so".format(module_path)):
            return name
    elif platform == "darwin":
        for name in glob.glob("{}/runtime.cpython-*-darwin.so".format(module_path)):
            return name
    else:
        return None


def _find_user_lib_path():
    libnux_path = None
    for lib_path in os.getenv('LD_LIBRARY_PATH').split(":"):
        if platform == "linux":
            for name in glob.glob("{}/libnux.so*".format(lib_path)):
                libnux_path = name
                break
        elif platform == "darwin":
            for name in glob.glob("{}/libnux.dylib*".format(lib_path)):
                libnux_path = name
                break

    return libnux_path


def _find_global_lib_path():
    return util.find_library("nux")


def find_native_lib_path():
    """Finding a native lib according to the priority
    1. If the environment variable 'LD_LIBRARY_PATH' is set,
    this function tries to find native library found from LD_LIBRARY_PATH.
    2. Otherwise, it tries to find the native library embedded in the python package.
    3. If the embedded native library cannot be found,
    it tries find the native library from global library paths, such as /usr/lib, /usr/local/lib.
    """
    libnux_path = None

    if os.getenv('LD_LIBRARY_PATH') is not None:
        libnux_path = _find_user_lib_path()

    if libnux_path is None:
        libnux_path = _find_local_lib_path()
        if libnux_path is None:
            libnux_path = _find_global_lib_path()

    if libnux_path is None:
        raise SystemExit('fail to find libnux')

    return libnux_path


def __register_common_capis(libnux):
    libnux.version.argtypes = []
    libnux.version.restype = c_char_p

    libnux.git_short_hash.argtypes = []
    libnux.git_short_hash.restype = c_char_p

    libnux.build_timestamp.argtypes = []
    libnux.build_timestamp.restype = c_char_p

    libnux.register_signal_handler.argtypes = []
    libnux.register_signal_handler.restype = None

    libnux.enable_logging.argtypes = [c_int]
    libnux.enable_logging.restype = None


def find_native_libs():
    """Finding a native lib according to the priority
    1. If the environment variable 'LD_LIBRARY_PATH' is set,
    this function tries to find native library found from LD_LIBRARY_PATH.
    2. Otherwise, it tries to find the native library embedded in the python package.
    3. If the embedded native library cannot be found,
    it tries find the native library from global library paths, such as /usr/lib, /usr/local/lib.
    """

    libnux_path = find_native_lib_path()
    libnux_ = CDLL(libnux_path)

    if libnux_ is not None:
        __register_common_capis(libnux_)

        LOG.info('loaded native library %s (%s %s)' % (
            libnux_path,
            libnux_.version().decode('utf-8'),
            libnux_.git_short_hash().decode('utf-8')), file=sys.stderr)
    else:
        raise SystemExit('fail to load native library')

    return libnux_


LIBNUX = find_native_libs()


def compiler_version() -> str:
    return f"{LIBNUX.version().decode('utf-8')} " \
           f"(rev: {LIBNUX.git_short_hash().decode('utf-8')} built at {LIBNUX.build_timestamp().decode('utf-8')})"


class NuxLogLevel(IntEnum):
    """Python object correspondnig to nux_log_level_t in Nux C API"""
    OFF = 0
    ERROR = 1
    WARN = 2
    INFO = 3