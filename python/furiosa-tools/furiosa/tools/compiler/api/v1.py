"""C Native library binding"""
import ctypes
from ctypes import CDLL, Structure, byref, c_bool, c_char_p, c_int, c_ulonglong, c_void_p
from enum import IntEnum
import logging
from pathlib import Path
import sys
from typing import Dict, Optional, Union

from furiosa.common.error import FuriosaError, is_err
from furiosa.common.native import LogLevel, find_native_libs

LOG = logging.getLogger(__name__)

DEFAULT_ENCODING = "utf-8"
DEFAULT_TARGET_NPU = "warboy-2pe"


class FcBuffer(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("len", c_ulonglong),
    ]


def __register_compiler_apis(ciface):
    ciface.fc_version.argtypes = []
    ciface.fc_version.restype = c_char_p

    ciface.fc_revision.argtypes = []
    ciface.fc_revision.restype = c_char_p

    ciface.fc_buildtime.argtypes = []
    ciface.fc_buildtime.restype = c_char_p

    ciface.fc_enable_logging.argtypes = [c_int]
    ciface.fc_enable_logging.restype = None

    ciface.register_signal_handler.argtypes = []
    ciface.register_signal_handler.restype = None

    # Making aliases to be compatible with find_native_libs
    ciface.version = ciface.fc_version
    ciface.git_short_hash = ciface.fc_revision


## Definition of Compiler Native C API
LIBCOMPILER = find_native_libs("furiosa_compiler", register_hook=__register_compiler_apis)
LIBCOMPILER.fc_compile.argtypes = [c_void_p, c_char_p, c_char_p, c_void_p, c_void_p]
LIBCOMPILER.fc_compile.restype = c_int

LIBCOMPILER.fc_destroy_buffer.argtypes = [c_void_p]
LIBCOMPILER.fc_destroy_buffer.restype = None


LIBCOMPILER.fc_create_options.argtypes = []
LIBCOMPILER.fc_create_options.restype = c_void_p

LIBCOMPILER.fc_destroy_options.argtypes = [c_void_p]
LIBCOMPILER.fc_destroy_options.restype = None

LIBCOMPILER.fc_options_target_ir.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.fc_options_target_ir.restype = None

LIBCOMPILER.fc_options_dot_graph.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.fc_options_dot_graph.restype = None

LIBCOMPILER.fc_options_memory_analysis.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.fc_options_memory_analysis.restype = None

LIBCOMPILER.fc_options_batch_size.argtypes = [c_void_p, c_ulonglong]
LIBCOMPILER.fc_options_batch_size.restype = None

LIBCOMPILER.fc_options_auto_batch_size.argtypes = [c_void_p, c_bool]
LIBCOMPILER.fc_options_auto_batch_size.restype = None

LIBCOMPILER.fc_options_split_after_lower.argtypes = [c_void_p, c_bool]
LIBCOMPILER.fc_options_split_after_lower.restype = None

LIBCOMPILER.fc_options_target_npu.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.fc_options_target_npu.restype = None

LIBCOMPILER.fc_options_ga_optimization.argtypes = [c_void_p, c_bool]
LIBCOMPILER.fc_options_ga_optimization.restype = None

LIBCOMPILER.fc_options_ga_population_size.argtypes = [c_void_p, c_ulonglong]
LIBCOMPILER.fc_options_ga_population_size.restype = None

LIBCOMPILER.fc_options_ga_generation_limit.argtypes = [c_void_p, c_ulonglong]
LIBCOMPILER.fc_options_ga_generation_limit.restype = None

LIBCOMPILER.fc_options_ga_nondeterministic.argtypes = [c_void_p, c_bool]
LIBCOMPILER.fc_options_ga_nondeterministic.restype = None

LIBCOMPILER.fc_options_ga_pin_tensors.argtypes = [c_void_p, c_bool]
LIBCOMPILER.fc_options_ga_pin_tensors.restype = None

LIBCOMPILER.fc_options_ga_init_tactic.argtypes = [c_void_p, c_char_p]
LIBCOMPILER.fc_options_ga_init_tactic.restype = None

LIBCOMPILER.fc_options_enable_cache.argtypes = [c_void_p, c_bool]
LIBCOMPILER.fc_options_enable_cache.restype = None

# Register Ctrl-C signal handler to interrupt native side for long running job
LIBCOMPILER.register_signal_handler()


class NativeError(IntEnum):
    """Python object correspondnig to nux_error_t in furiosa-libcompiler C API"""

    SUCCESS = 0
    COMPILATION_ERROR = 1
    IO_ERROR = 2
    CONFIGURATION_MISMATCH = 3
    INVALID_FORMAT = 4
    INVALID_MODEL = 5
    INVALID_NPUID = 6
    OUT_OF_DRAM = 7
    OUT_OF_SRAM = 8
    OUT_OF_INSTUCTION_MEMORY = 9
    TFLITE_TO_DFG = 10
    USER_COMMAND = 11
    INVALID_CONFIG = 12
    OTHER = 13


class CompilerApiError(FuriosaError):
    def __init__(self, message: str, err_code: Optional[NativeError] = None):
        self.native_err = err_code
        super().__init__(message)


class InvalidTargetIrException(CompilerApiError):
    def __init__(self, format: str = None):
        super().__init__(f"invalid target ir '{format}'", NativeError.INVALID_FORMAT)


def __set_ga_param(options, key: str, value: object):
    if key.lower() == "population_size":
        if isinstance(value, int):
            LIBCOMPILER.fc_options_ga_population_size(options, value)
        else:
            raise CompilerApiError("population_size must be a positive integer")

    elif key.lower() == "generation_limit":
        if isinstance(value, int):
            LIBCOMPILER.fc_options_ga_generation_limit(options, value)
        else:
            CompilerApiError("generation_limit must be a positive integer")

    elif key.lower() == "max_prefetch_size":
        if isinstance(value, int):
            LIBCOMPILER.fc_options_ga_max_prefetch_size(options, value)
        else:
            raise CompilerApiError("max_prefetch_size must be a positive integer")

    elif key.lower() == "nondeterministic":
        if isinstance(value, bool):
            LIBCOMPILER.fc_options_ga_nondeterministic(options, value)
        else:
            raise CompilerApiError("nondeterministic must be a boolean value")

    elif key.lower() == "pin_tensors":
        if isinstance(value, bool):
            LIBCOMPILER.fc_options_ga_pin_tensors(options, value)
        else:
            raise CompilerApiError("pin_tensors must be a boolean value")

    elif key.lower() == "init_tactic":
        if isinstance(value, str) and value.lower() in ['random', 'heuristic']:
            LIBCOMPILER.fc_options_ga_init_tactic(options, value.encode(DEFAULT_ENCODING))
        else:
            raise CompilerApiError("init_tactic must be either 'random' or 'heuristic'")

    else:
        raise CompilerApiError(f"unknown genetic algorithm parameter: '{key}'")


class VersionInfo:
    def __init__(self):
        self.version = f"{LIBCOMPILER.fc_version().decode(DEFAULT_ENCODING)}"
        self.git_hash = f"{LIBCOMPILER.fc_revision().decode(DEFAULT_ENCODING)}"
        self.build_timestamp = f"{LIBCOMPILER.fc_buildtime().decode(DEFAULT_ENCODING)}"


def version_string() -> str:
    info = VersionInfo()
    return f"{info.version} " f"(rev: {info.git_hash} " f"built at {info.build_timestamp})"


def __check_target_ir(target_ir: str):
    if not target_ir.lower().strip() in ["dfg", "ldfg", "cdfg", "gir", "sir", "lir", "enf"]:
        raise InvalidTargetIrException(target_ir)


def compile(
    input_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    target_ir: str = "enf",
    dot_graph: Optional[Union[str, Path]] = None,
    analyze_memory: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    split_after_lower: Optional[bool] = None,
    auto_batch_size: Optional[bool] = None,
    ga_params: Optional[Dict[str, str]] = None,
    target_npu: str = DEFAULT_TARGET_NPU,
    cache_enabled: bool = True,
    verbose: bool = False,
) -> int:
    if verbose:
        LIBCOMPILER.fc_enable_logging(LogLevel.INFO)

    input_bytes = Path(input_path).read_bytes()
    input_buf = FcBuffer(ctypes.cast(input_bytes, c_void_p).value, len(input_bytes))

    __check_target_ir(target_ir)

    options = LIBCOMPILER.fc_create_options()

    LIBCOMPILER.fc_options_target_ir(options, target_ir.encode(DEFAULT_ENCODING))
    LIBCOMPILER.fc_options_target_npu(options, target_npu.encode(DEFAULT_ENCODING))
    LIBCOMPILER.fc_options_enable_cache(options, cache_enabled)

    if analyze_memory:
        LIBCOMPILER.fc_options_memory_analysis(
            options, str(analyze_memory).encode(DEFAULT_ENCODING)
        )
    if dot_graph:
        LIBCOMPILER.fc_options_dot_graph(options, str(dot_graph).encode(DEFAULT_ENCODING))
    if batch_size:
        LIBCOMPILER.fc_options_batch_size(options, batch_size)
    if auto_batch_size:
        LIBCOMPILER.fc_options_auto_batch_size(options, auto_batch_size)
    if split_after_lower:
        LIBCOMPILER.fc_options_split_after_lower(options, split_after_lower)
    if ga_params:
        LIBCOMPILER.fc_options_ga_optimization(options, True)
        for key in ga_params.keys():
            value = ga_params[key]
            __set_ga_param(options, key, value)

    output_buf = FcBuffer()
    errno = LIBCOMPILER.fc_compile(options, None, None, byref(input_buf), byref(output_buf))

    if is_err(errno):
        # it's ok because furiosa-compiler will print out the error message to stderr
        return errno.value if isinstance(errno, ctypes.c_int) else errno

    try:
        with open(output_path, "wb") as output:
            array_type = ctypes.c_ubyte * output_buf.len
            buffer = array_type.from_address(output_buf.data)
            output.write(buffer)
            print(f"The output has been saved to {output_path}", file=sys.stderr)
    finally:
        LIBCOMPILER.fc_destroy_buffer(byref(output_buf))

    # Happy return
    return 0
