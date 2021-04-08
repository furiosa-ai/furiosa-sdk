"""C Native library binding"""

import ctypes
import glob
import logging
import os
from ctypes import CDLL, c_void_p, c_char_p, POINTER, c_ulonglong, c_bool, c_int, util
from sys import platform

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def _find_local_libs():
    module_path = os.path.dirname(__file__)
    if platform == "linux":
        for name in glob.glob("{}/runtime.cpython-*.so".format(module_path)):
            return name
    elif platform == "darwin":
        for name in glob.glob("{}/runtime.cpython-*-darwin.so".format(module_path)):
            return name
    else:
        return None


def _find_user_libs():
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


def _find_global_libs():
    return util.find_library("nux")


def _find_native_libs():
    """Finding a native lib according to the priority
    1. If the environment variable 'LD_LIBRARY_PATH' is set,
    this function tries to find native library found from LD_LIBRARY_PATH.
    2. Otherwise, it tries to find the native library embedded in the python package.
    3. If the embedded native library cannot be found,
    it tries find the native library from global library paths, such as /usr/lib, /usr/local/lib.
    """

    libnux_path = None

    if os.getenv('LD_LIBRARY_PATH') is not None:
        libnux_path = _find_user_libs()

    if libnux_path is None:
        libnux_path = _find_local_libs()
        if libnux_path is None:
            libnux_path = _find_global_libs()

    if libnux_path is None:
        raise SystemExit('fail to find libnux')

    libnux_ = CDLL(libnux_path)

    if libnux_ is not None:
        LOG.info('successfully loaded dynamic library %s', libnux_path)
    else:
        raise SystemExit('fail to load native library')

    return libnux_


LIBNUX = _find_native_libs()

## Definition of Native C Foreign Functions
LIBNUX.version.argtypes = []
LIBNUX.version.restype = c_char_p

LIBNUX.git_short_hash.argtypes = []
LIBNUX.git_short_hash.restype = c_char_p

LIBNUX.build_timestamp.argtypes = []
LIBNUX.build_timestamp.restype = c_char_p

LIBNUX.enable_furiosa_logging.argtypes = []
LIBNUX.enable_furiosa_logging.restype = None

LIBNUX.nux_session_option_create.argtypes = []
LIBNUX.nux_session_option_create.restype = c_void_p

LIBNUX.nux_session_option_set_device.argtypes = [c_void_p, c_char_p]
LIBNUX.nux_session_option_set_device.restype = None

LIBNUX.nux_session_option_set_compiler_config.argtypes = [c_void_p, c_char_p]
LIBNUX.nux_session_option_set_compiler_config.restype = c_int

LIBNUX.nux_input_num.argtypes = [c_void_p]
LIBNUX.nux_input_num.restype = c_int

LIBNUX.nux_output_num.argtypes = [c_void_p]
LIBNUX.nux_output_num.restype = c_int

LIBNUX.nux_input_desc.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_input_desc.restype = c_void_p

LIBNUX.nux_output_desc.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_output_desc.restype = c_void_p

LIBNUX.nux_tensor_dim_num.argtypes = [c_void_p]
LIBNUX.nux_tensor_dim_num.restype = c_ulonglong

LIBNUX.nux_tensor_dim.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_tensor_dim_num.restype = c_ulonglong

LIBNUX.nux_tensor_axis.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_tensor_axis.restype = c_int

LIBNUX.nux_tensor_dtype.argtypes = [c_void_p]
LIBNUX.nux_tensor_dtype.restype = c_int

LIBNUX.nux_session_create.argtypes = [c_void_p, c_ulonglong, c_void_p, POINTER(c_void_p)]
LIBNUX.nux_session_create.restype = c_int

LIBNUX.nux_session_get_model.argtypes = [c_void_p]
LIBNUX.nux_session_get_model.restype = c_void_p

LIBNUX.nux_session_run.argtypes = [c_void_p, c_void_p, c_void_p]
LIBNUX.nux_session_run.restype = c_int

LIBNUX.nux_session_destroy.argtypes = [c_void_p]
LIBNUX.nux_session_destroy.restype = None

LIBNUX.nux_async_session_create.argtypes = \
    [c_void_p, c_ulonglong, c_void_p, POINTER(c_void_p), POINTER(c_void_p)]
LIBNUX.nux_async_session_create.restype = c_int

LIBNUX.nux_async_session_get_model.argtypes = [c_void_p]
LIBNUX.nux_async_session_get_model.restype = c_void_p

LIBNUX.nux_async_session_run.argtypes = [c_void_p, ctypes.py_object, c_void_p]
LIBNUX.nux_async_session_run.restype = c_int

LIBNUX.nux_async_session_destroy.argtypes = [c_void_p]
LIBNUX.nux_async_session_destroy.restype = None

LIBNUX.nux_completion_queue_next.argtypes = [c_void_p, POINTER(ctypes.py_object), POINTER(c_void_p)]
LIBNUX.nux_completion_queue_next.restype = c_bool

LIBNUX.nux_tensor_array_create_inputs.argtypes = [c_void_p]
LIBNUX.nux_tensor_array_create_inputs.restype = c_void_p

LIBNUX.nux_tensor_array_create_outputs.argtypes = [c_void_p]
LIBNUX.nux_tensor_array_create_outputs.restype = c_void_p

LIBNUX.nux_tensor_buffer_size.argtypes = [c_void_p]

LIBNUX.nux_tensor_array_len.argtypes = [c_void_p]
LIBNUX.nux_tensor_array_len.restype = c_ulonglong

LIBNUX.nux_tensor_array_get.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_tensor_array_get.restype = c_void_p

LIBNUX.nux_tensor_array_destroy.argtypes = [c_void_p]
LIBNUX.nux_tensor_array_destroy.restype = None

LIBNUX.tensor_set_buffer.argtypes = [c_void_p, POINTER(ctypes.c_uint8), c_ulonglong]
LIBNUX.tensor_set_buffer.restype = c_int

LIBNUX.tensor_fill_buffer.argtypes = [c_void_p, POINTER(ctypes.c_uint8), c_ulonglong]
LIBNUX.tensor_fill_buffer.restype = c_int

LIBNUX.nux_tensor_len.argtypes = [c_void_p]
LIBNUX.nux_tensor_len.restype = c_ulonglong

LIBNUX.nux_tensor_size.argtypes = [c_void_p]
LIBNUX.nux_tensor_size.restype = c_ulonglong

LIBNUX.tensor_get_buffer.argtypes = \
    [c_void_p, POINTER(POINTER(ctypes.c_uint8)), POINTER(c_ulonglong)]
LIBNUX.tensor_get_buffer.restype = c_int

# To control manually the reference count
increase_ref_count = ctypes.pythonapi.Py_IncRef
increase_ref_count.argtypes = [ctypes.py_object]
increase_ref_count.restype = None

decref = ctypes.pythonapi.Py_DecRef
decref.argtypes = [ctypes.py_object]
decref.restype = None

# Enable FUriosa logger
LIBNUX.enable_furiosa_logging()
