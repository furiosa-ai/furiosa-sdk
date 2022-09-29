"""C Native library binding"""

import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_int, c_ulonglong, c_void_p
import logging
import os
from typing import List

from furiosa.common.native import LogLevel, find_native_libs
from furiosa.runtime import consts

LOG = logging.getLogger(__name__)


def _nux_log_level_from_env() -> int:
    level = os.environ.get(consts.ENV_FURIOSA_LOG_LEVEL, 'INFO')
    return LogLevel[level.upper()].value


def convert_to_cchar_array(list: List[str]):
    """
    Convert List[str] to *const *const char of ctypes.

    This function creates an array of POINTER(c_char_p) whose length is
    len(list) + 1, and fill the array of pointers with bytes. The reason
    why we append one more element is to add a null pointer to the end
    of the list. Then, C, C++ side will be able to recognize where is
    the end of the list without length.
    """
    bytes_list = [bytes(str, 'utf-8') for str in list]
    # create an array of ctypes. Please refer to https://docs.python.org/3/library/ctypes.html#arrays.
    ptrs_list = (ctypes.c_char_p * (len(bytes_list) + 1))()
    ptrs_list[:-1] = bytes_list
    return ptrs_list


# Definition of Session Native C API
LIBNUX = find_native_libs("nux")

LIBNUX.nux_session_option_create.argtypes = []
LIBNUX.nux_session_option_create.restype = c_void_p

LIBNUX.nux_session_option_set_device.argtypes = [c_void_p, c_char_p]
LIBNUX.nux_session_option_set_device.restype = None

LIBNUX.nux_session_option_set_compiler_config.argtypes = [c_void_p, c_char_p]
LIBNUX.nux_session_option_set_compiler_config.restype = c_int

LIBNUX.nux_session_option_set_compiler_log_path.argtypes = [c_void_p, c_char_p]
LIBNUX.nux_session_option_set_compiler_log_path.restype = None

LIBNUX.nux_session_option_set_input_queue_size.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_session_option_set_input_queue_size.restype = None

LIBNUX.nux_session_option_set_output_queue_size.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_session_option_set_output_queue_size.restype = None

LIBNUX.nux_session_option_set_worker_num.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_session_option_set_worker_num.restype = None

LIBNUX.nux_session_option_set_batch_size.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_session_option_set_batch_size.restype = None

LIBNUX.nux_session_option_enable_compiler_hints.argtypes = [c_void_p, c_bool]
LIBNUX.nux_session_option_enable_compiler_hints.restype = None

LIBNUX.nux_session_option_destroy.argtypes = [c_void_p]
LIBNUX.nux_session_option_destroy.restype = None

LIBNUX.nux_input_num.argtypes = [c_void_p]
LIBNUX.nux_input_num.restype = c_int

LIBNUX.nux_output_num.argtypes = [c_void_p]
LIBNUX.nux_output_num.restype = c_int

LIBNUX.nux_input_desc.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_input_desc.restype = c_void_p

LIBNUX.nux_output_desc.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_output_desc.restype = c_void_p

LIBNUX.nux_tensor_name.argtypes = [c_void_p]
LIBNUX.nux_tensor_name.restype = c_void_p

LIBNUX.nux_tensor_dim_num.argtypes = [c_void_p]
LIBNUX.nux_tensor_dim_num.restype = c_ulonglong

LIBNUX.nux_tensor_dim.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_tensor_dim_num.restype = c_ulonglong

LIBNUX.nux_tensor_stride.argtypes = [c_void_p, c_ulonglong]
LIBNUX.nux_tensor_stride.restype = c_ulonglong

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

LIBNUX.nux_session_run_with.argtypes = [
    c_void_p,
    POINTER(c_char_p),
    c_ulonglong,
    POINTER(c_char_p),
    c_ulonglong,
    c_void_p,
    c_void_p,
]
LIBNUX.nux_session_run_with.restype = c_int

LIBNUX.nux_session_destroy.argtypes = [c_void_p]
LIBNUX.nux_session_destroy.restype = None

LIBNUX.nux_async_session_create.argtypes = [
    c_void_p,
    c_ulonglong,
    c_void_p,
    POINTER(c_void_p),
    POINTER(c_void_p),
]
LIBNUX.nux_async_session_create.restype = c_int

LIBNUX.nux_async_session_get_model.argtypes = [c_void_p]
LIBNUX.nux_async_session_get_model.restype = c_void_p

LIBNUX.nux_async_session_run.argtypes = [c_void_p, ctypes.py_object, c_void_p]
LIBNUX.nux_async_session_run.restype = c_int

LIBNUX.nux_async_session_destroy.argtypes = [c_void_p]
LIBNUX.nux_async_session_destroy.restype = None

LIBNUX.nux_completion_queue_next.argtypes = [
    c_void_p,
    POINTER(ctypes.py_object),
    POINTER(c_void_p),
    POINTER(c_int),
]
LIBNUX.nux_completion_queue_next.restype = c_bool

LIBNUX.nux_completion_queue_next_timeout.argtypes = [
    c_void_p,
    c_ulonglong,
    POINTER(ctypes.py_object),
    POINTER(c_void_p),
    POINTER(c_int),
]
LIBNUX.nux_completion_queue_next_timeout.restype = c_bool

LIBNUX.nux_tensor_array_create_by_names.argtypes = [c_void_p, POINTER(ctypes.c_char_p), c_ulonglong]
LIBNUX.nux_tensor_array_create_by_names.restype = c_void_p

LIBNUX.nux_tensor_array_allocate_by_names.argtypes = [
    c_void_p,
    POINTER(ctypes.c_char_p),
    c_ulonglong,
]
LIBNUX.nux_tensor_array_allocate_by_names.restype = c_void_p

LIBNUX.nux_tensor_array_create_inputs.argtypes = [c_void_p]
LIBNUX.nux_tensor_array_create_inputs.restype = c_void_p

LIBNUX.nux_tensor_array_allocate_inputs.argtypes = [c_void_p]
LIBNUX.nux_tensor_array_allocate_inputs.restype = c_void_p

LIBNUX.nux_tensor_array_create_outputs.argtypes = [c_void_p]
LIBNUX.nux_tensor_array_create_outputs.restype = c_void_p

LIBNUX.nux_tensor_array_allocate_outputs.argtypes = [c_void_p]
LIBNUX.nux_tensor_array_allocate_outputs.restype = c_void_p

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

LIBNUX.tensor_get_buffer.argtypes = [
    c_void_p,
    POINTER(POINTER(ctypes.c_uint8)),
    POINTER(c_ulonglong),
]
LIBNUX.tensor_get_buffer.restype = c_int

LIBNUX.nux_buffer_destroy.argtypes = [POINTER(ctypes.c_uint8), c_ulonglong]
LIBNUX.nux_buffer_destroy.restype = None

LIBNUX.nux_string_destroy.argtypes = [c_void_p]
LIBNUX.nux_string_destroy.restype = None

LIBNUX.profiler_enable.argtypes = [c_int, c_int, c_char_p, POINTER(c_void_p)]
LIBNUX.profiler_enable.restype = c_int

LIBNUX.profiler_disable.argtypes = [c_void_p]
LIBNUX.profiler_disable.restype = None

LIBNUX.profiler_record_start.argtypes = [c_char_p, c_bool]
LIBNUX.profiler_record_start.restype = POINTER(c_void_p)

LIBNUX.profiler_record_end.argtypes = [POINTER(c_void_p)]
LIBNUX.profiler_record_end.restype = None

# To control manually the reference count
increase_ref_count = ctypes.pythonapi.Py_IncRef
increase_ref_count.argtypes = [ctypes.py_object]
increase_ref_count.restype = None

decref = ctypes.pythonapi.Py_DecRef
decref.argtypes = [ctypes.py_object]
decref.restype = None

# Enable Furiosa logger
LIBNUX.enable_logging(_nux_log_level_from_env())

# Register Ctrl-C signal handler to interrupt native side for long running job
LIBNUX.register_signal_handler()


def runtime_version() -> str:
    return (
        f"{LIBNUX.version().decode('utf-8')} "
        f"(rev: {LIBNUX.git_short_hash().decode('utf-8')} built at {LIBNUX.build_timestamp().decode('utf-8')})"
    )
