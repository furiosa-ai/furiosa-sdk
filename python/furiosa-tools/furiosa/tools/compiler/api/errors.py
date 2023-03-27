import ctypes
from enum import IntEnum
from typing import Optional

from furiosa.common.error import FuriosaError


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
    OTHER = 255


class NativeException(FuriosaError):
    _native_err: Optional[NativeError]

    def __init__(self, message: str, native_err: NativeError = None):
        self._native_err = native_err
        super().__init__(message)

    def native_error(self) -> Optional[NativeError]:
        return self._native_err

    def __repr__(self):
        if self._native_err is None:
            return self._message

        return f'{self._message} (native error code: {self._native_err})'

    def __str__(self):
        return self.__repr__()


class CompilerApiError(FuriosaError):
    def __init__(self, message: str, err_code: Optional[NativeError] = None):
        self.native_err = err_code
        super().__init__(message)


class InvalidTargetIrException(CompilerApiError):
    def __init__(self, format: str = None):
        super().__init__(f"invalid target ir '{format}'", NativeError.INVALID_FORMAT)


class CompilationError(NativeException):
    def __init__(self):
        super().__init__("Compilation error", NativeError.COMPILATION_ERROR)


class IOError(NativeException):
    def __init__(self):
        super().__init__("IO error", NativeError.IO_ERROR)


class ConfigurationMismatch(NativeException):
    def __init__(self):
        super().__init__("Configuration mismatch", NativeError.CONFIGURATION_MISMATCH)


class InvalidFormat(NativeException):
    def __init__(self):
        super().__init__("Invalid format", NativeError.INVALID_FORMAT)


class InvalidModel(NativeException):
    def __init__(self):
        super().__init__("Invalid model", NativeError.INVALID_MODEL)


class InvalidNPUId(NativeException):
    def __init__(self):
        super().__init__("Invalid NPU id", NativeError.INVALID_NPUID)


class OutOfDram(NativeException):
    def __init__(self):
        super().__init__("Out of dram", NativeError.OUT_OF_DRAM)


class OutOfSram(NativeException):
    def __init__(self):
        super().__init__("Out of sram", NativeError.OUT_OF_SRAM)


class OutOfInstructionMemory(NativeException):
    def __init__(self):
        super().__init__("Out of instruction memory", NativeError.OUT_OF_INSTUCTION_MEMORY)


class TfliteToDFG(NativeException):
    def __init__(self):
        super().__init__("TFLITE to dfg", NativeError.TFLITE_TO_DFG)


class UserCommand(NativeException):
    def __init__(self):
        super().__init__("User command", NativeError.USER_COMMAND)


class InvalidConfig(NativeException):
    def __init__(self):
        super().__init__("Invalid config", NativeError.INVALID_CONFIG)


class InternalError(NativeException):
    def __init__(self):
        super().__init__("Unknown", NativeError.OTHER)


_errors_to_exceptions = {
    NativeError.COMPILATION_ERROR: CompilationError(),
    NativeError.IO_ERROR: IOError(),
    NativeError.CONFIGURATION_MISMATCH: ConfigurationMismatch(),
    NativeError.INVALID_FORMAT: InvalidFormat(),
    NativeError.INVALID_MODEL: InvalidModel(),
    NativeError.INVALID_NPUID: InvalidNPUId(),
    NativeError.OUT_OF_DRAM: OutOfDram(),
    NativeError.OUT_OF_SRAM: OutOfSram(),
    NativeError.OUT_OF_INSTUCTION_MEMORY: OutOfInstructionMemory(),
    NativeError.TFLITE_TO_DFG: TfliteToDFG(),
    NativeError.USER_COMMAND: UserCommand(),
    NativeError.INVALID_CONFIG: InvalidConfig(),
    NativeError.OTHER: InternalError(),
}


def into_exception(err):
    err = err.value if isinstance(err, ctypes.c_int) else err

    if err == NativeError.SUCCESS:
        return RuntimeError(msg='CompilerErr.SUCCESS cannot be CompilerException')

    if err in _errors_to_exceptions:
        return _errors_to_exceptions[err]

    return InternalError()
