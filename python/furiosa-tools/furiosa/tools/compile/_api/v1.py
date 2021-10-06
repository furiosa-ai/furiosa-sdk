"""C Native library binding"""

import ctypes
import glob
import logging
import os
from ctypes import CDLL, POINTER, c_bool, c_char_p, c_int, c_ulonglong, c_void_p, util
from enum import IntEnum
from sys import platform

from furiosa.common.native import LIBNUX, NuxLogLevel, compiler_version


## Definition of Compiler Native C API
LIBCOMPILER = LIBNUX
LIBCOMPILER.compiler_run.argtypes = [c_void_p]
LIBCOMPILER.compiler_run.restype = c_int

LIBCOMPILER.compiler_options_create.argtypes = []
LIBCOMPILER.compiler_options_create.restype = c_void_p

LIBCOMPILER.compiler_options_destroy.argtypes = [c_void_p]
LIBCOMPILER.compiler_options_destroy.restype = None

LIBCOMPILER.compiler_options_input.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.compiler_options_input.restype = None

LIBCOMPILER.compiler_options_output.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.compiler_options_output.restype = None

LIBCOMPILER.compiler_options_target_ir.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.compiler_options_target_ir.restype = None

LIBCOMPILER.compiler_options_dot_graph.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.compiler_options_dot_graph.restype = None

LIBCOMPILER.compiler_options_memory_analysis.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.compiler_options_memory_analysis.restype = None

LIBCOMPILER.compiler_options_batch_size.argtypes = [c_void_p, c_ulonglong]
LIBCOMPILER.compiler_options_batch_size.restype = None

LIBCOMPILER.compiler_options_auto_batch_size.argtypes = [c_void_p, c_bool]
LIBCOMPILER.compiler_options_auto_batch_size.restype = None

LIBCOMPILER.compiler_options_split_after_lower.argtypes = [c_void_p, c_bool]
LIBCOMPILER.compiler_options_split_after_lower.restype = None

LIBCOMPILER.compiler_options_target_npu.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.compiler_options_target_npu.restype = None

LIBCOMPILER.compiler_options_ga_optimization.argtypes = [c_void_p, c_bool]
LIBCOMPILER.compiler_options_ga_optimization.restype = None

LIBCOMPILER.compiler_options_ga_population_size.argtypes = [c_void_p, c_ulonglong]
LIBCOMPILER.compiler_options_ga_population_size.restype = None

LIBCOMPILER.compiler_options_ga_generation_limit.argtypes = [c_void_p, c_ulonglong]
LIBCOMPILER.compiler_options_ga_generation_limit.restype = None

LIBCOMPILER.compiler_options_ga_nondeterministic.argtypes = [c_void_p, c_bool]
LIBCOMPILER.compiler_options_ga_nondeterministic.restype = None

LIBCOMPILER.compiler_options_ga_pin_tensors.argtypes = [c_void_p, c_bool]
LIBCOMPILER.compiler_options_ga_pin_tensors.restype = None

LIBCOMPILER.compiler_options_ga_init_tactic.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.compiler_options_ga_init_tactic.restype = None


# Register Ctrl-C signal handler to interrupt native side for long running job
LIBCOMPILER.register_signal_handler()
