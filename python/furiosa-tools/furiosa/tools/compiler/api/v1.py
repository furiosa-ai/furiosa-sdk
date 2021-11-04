"""C Native library binding"""

from ctypes import c_bool, c_char_p, c_int, c_ulonglong, c_void_p
from pathlib import Path
from typing import Dict, Optional, Union

from furiosa.common.native import LogLevel, find_native_libs

DEFAULT_ENCODING = "utf-8"
DEFAULT_NPU = "warboy"


## Definition of Compiler Native C API
LIBCOMPILER = find_native_libs("nux")
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


class CompilerApiError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__()


def __set_ga_param(options, key: str, value: object):
    if key.lower() == "population_size":
        if isinstance(value, int):
            LIBCOMPILER.compiler_options_ga_population_size(options, value)
        else:
            raise CompilerApiError("population_size must be a positive integer")

    elif key.lower() == "generation_limit":
        if isinstance(value, int):
            LIBCOMPILER.compiler_options_ga_generation_limit(options, value)
        else:
            CompilerApiError("generation_limit must be a positive integer")

    elif key.lower() == "max_prefetch_size":
        if isinstance(value, int):
            LIBCOMPILER.compiler_options_ga_max_prefetch_size(options, value)
        else:
            raise CompilerApiError("max_prefetch_size must be a positive integer")

    elif key.lower() == "nondeterministic":
        if isinstance(value, bool):
            LIBCOMPILER.compiler_options_ga_nondeterministic(options, value)
        else:
            raise CompilerApiError("nondeterministic must be a boolean value")

    elif key.lower() == "pin_tensors":
        if isinstance(value, bool):
            LIBCOMPILER.compiler_options_ga_pin_tensors(options, value)
        else:
            raise CompilerApiError("pin_tensors must be a boolean value")

    elif key.lower() == "init_tactic":
        if isinstance(value, str) and value.lower() in ['random', 'heuristic']:
            LIBCOMPILER.compiler_options_ga_init_tactic(options, value.encode(DEFAULT_ENCODING))
        else:
            raise CompilerApiError("init_tactic must be either 'random' or 'heuristic'")

    else:
        raise CompilerApiError(f"unknown genetic algorithm parameter: '{key}'")


class VersionInfo:
    def __init__(self):
        self.version = f"{LIBCOMPILER.version().decode(DEFAULT_ENCODING)}"
        self.git_hash = f"{LIBCOMPILER.git_short_hash().decode(DEFAULT_ENCODING)}"
        self.build_timestamp = f"{LIBCOMPILER.build_timestamp().decode(DEFAULT_ENCODING)}"


def version_string() -> str:
    info = VersionInfo()
    return f"{info.version} " f"(rev: {info.git_hash} " f"built at {info.build_timestamp})"


def compile(
    input: Union[str, Path],
    output: Union[str, Path] = None,
    target_ir: str = "enf",
    dot_graph: Optional[Union[str, Path]] = None,
    analyze_memory: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    split_after_lower: Optional[bool] = None,
    auto_batch_size: Optional[bool] = None,
    ga_params: Optional[Dict[str, str]] = None,
    target_npu: str = DEFAULT_NPU,
    verbose: bool = False,
) -> int:

    if verbose:
        LIBCOMPILER.enable_logging(LogLevel.INFO)

    options = LIBCOMPILER.compiler_options_create()

    LIBCOMPILER.compiler_options_input(options, str(input).encode(DEFAULT_ENCODING))

    if output:
        output_path = str(output)
    else:
        output_path = f"output.{target_ir}"

    LIBCOMPILER.compiler_options_output(options, output_path.encode(DEFAULT_ENCODING))
    LIBCOMPILER.compiler_options_target_ir(options, target_ir.encode(DEFAULT_ENCODING))
    LIBCOMPILER.compiler_options_target_npu(options, target_npu.encode(DEFAULT_ENCODING))

    if analyze_memory:
        LIBCOMPILER.compiler_options_memory_analysis(
            options, str(analyze_memory).encode(DEFAULT_ENCODING)
        )
    if dot_graph:
        LIBCOMPILER.compiler_options_dot_graph(options, str(dot_graph).encode(DEFAULT_ENCODING))
    if batch_size:
        LIBCOMPILER.compiler_options_batch_size(options, batch_size)
    if auto_batch_size:
        LIBCOMPILER.compiler_options_auto_batch_size(options, auto_batch_size)
    if split_after_lower:
        LIBCOMPILER.compiler_options_split_after_lower(options, split_after_lower)
    if ga_params:
        LIBCOMPILER.compiler_options_ga_optimization(options, True)
        for key in ga_params.keys():
            value = ga_params[key]
            __set_ga_param(options, key, value)

    return LIBCOMPILER.compiler_run(options)
