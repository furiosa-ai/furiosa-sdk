"""Nux Exception and Error"""
import ctypes
from enum import IntEnum
import typing
from typing import Optional

from furiosa.common.error import FuriosaError, is_err, is_ok


class NativeError(IntEnum):
    """Python object correspondnig to nux_error_t in Nux C API"""

    SUCCESS = 0
    NUX_CREATION_FAILED = 1
    MODEL_DEPLOY_FAILED = 2
    MODEL_EXECUTION_FAILED = 3
    INVALID_INPUT_INDEX = 4
    INVALID_OUTPUT_INDEX = 5
    INVALID_BUFFER = 6
    INVALID_INPUTS = 7
    INVALID_OUTPUTS = 8
    GET_TASK_FAILED = 9
    DUMP_PROFILE_FAILED = 10
    QUEUE_WAIT_TIMEOUT = 11
    QUEUE_NO_DATA = 12
    INCOMPATIBLE_MODEL = 13
    COMPILATION_FAILED = 14
    INTERNAL_ERROR = 15
    INVALID_YAML = 16
    INCOMPATIBLE_API_CLIENT_ERROR = 17
    API_CLIENT_INIT_FAILED = 18
    NO_API_KEY = 19
    NULL_POINTER_EXCEPTION = 20
    INVALID_SESSION_OPTIONS = 21
    SESSION_TERMINATED = 22
    DEVICE_BUSY = 23
    TENSOR_NAME_NOT_FOUND = 24
    UNSUPPORTED_FEATURE = 25
    INVALID_COMPILER_CONFIG = 30


class NativeException(FuriosaError):
    """general exception caused by Nuxpy"""

    _native_err: Optional[NativeError]

    def __init__(self, message: str, native_err: NativeError = None):
        self._native_err = native_err
        super().__init__(message)

    def native_error(self) -> Optional[NativeError]:
        """Return a native error if this exception comes from C native extension"""
        return self._native_err

    def __repr__(self):
        if self._native_err is None:
            return self._message

        return f'{self._message} (native error code: {self._native_err})'

    def __str__(self):
        return self.__repr__()


class IncompatibleModel(NativeException):
    """When Renegade compiler cannot recognize a given model image binary"""

    def __init__(self):
        super().__init__("model binary is not compatible", NativeError.INCOMPATIBLE_MODEL)


class CompilationFailed(NativeException):
    """when Nux fails to compile a given model image to NPU model binary"""

    def __init__(self):
        super().__init__(
            "fail to compile a given model to NPU binary", NativeError.COMPILATION_FAILED
        )


class InternalError(NativeException):
    """internal error or no corresponding error in Python binding"""

    def __init__(self, cause='unknown'):
        super().__init__("{}".format(cause), NativeError.INTERNAL_ERROR)


class UnsupportedTensorType(NativeException):
    """Unsupported tensor type"""

    def __init__(self):
        super().__init__(
            "numpy.ndarray, TensorArray are only supported", NativeError.INVALID_INPUTS
        )


class UnsupportedDataType(NativeException):
    """Unsupported tensor data type"""

    def __init__(self, dtype):
        super().__init__("unknown data type: {}".format(dtype))


class IncompatibleApiClientError(NativeException):
    """When both API client and server are incompatible"""

    def __init__(self):
        super().__init__(
            "incompatible client with Furiosa API server", NativeError.INCOMPATIBLE_API_CLIENT_ERROR
        )


class InvalidYamlException(NativeException):
    """When Renegade compiler cannot recognize a given model image binary"""

    def __init__(self):
        super().__init__("compiler config is not valid YAML", NativeError.INVALID_YAML)


class ApiClientInitFailed(NativeException):
    """when api client fails to initialize due to api keys or others"""

    def __init__(self):
        super().__init__("fail to initialize API client", NativeError.API_CLIENT_INIT_FAILED)


class NoApiKeyException(NativeException):
    """when api client fails to initialize due to api keys or others"""

    def __init__(self):
        super().__init__("no API keys. Please check your API keys.", NativeError.NO_API_KEY)


class InvalidSessionOption(NativeException):
    """when api client fails to initialize due to api keys or others"""

    def __init__(self):
        super().__init__(
            "invalid options passed to session.create() or create_async()",
            NativeError.INVALID_SESSION_OPTIONS,
        )


class QueueWaitTimeout(NativeException):
    """Timed out in Completion queue"""

    def __init__(self):
        super().__init__("queue waiting timed out", NativeError.QUEUE_WAIT_TIMEOUT)


class SessionTerminated(NativeException):
    """Session is already terminated"""

    def __init__(self):
        super().__init__("Session or AsyncSession terminated", NativeError.SESSION_TERMINATED)


class DeviceBusy(NativeException):
    """The device is already occupied"""

    def __init__(self):
        super().__init__("NPU device busy", NativeError.DEVICE_BUSY)


class InvalidInput(FuriosaError):
    """When input tensors are invalid with any reason"""

    def __init__(self, message: str = "Invalid input tensors"):
        super().__init__(message)


class TensorNameNotFound(NativeException):
    """When a given tensor name is not found in this model"""

    def __init__(self):
        super().__init__("Tensor name not found", NativeError.TENSOR_NAME_NOT_FOUND)


class UnsupportedFeature(NativeException):
    """Feature is not supported"""

    def __init__(self):
        super().__init__("Unsupported feature", NativeError.UNSUPPORTED_FEATURE)


class InvalidCompilerConfig(NativeException):
    """Compiler config is invalid"""

    def __init__(self):
        super().__init__("Invalid compiler config", NativeError.INVALID_COMPILER_CONFIG)


class SessionClosed(FuriosaError):
    """Session is already terminated"""

    def __init__(self):
        super().__init__("Session or AsyncSession is already closed. Please create a new session.")


_errors_to_exceptions = {
    NativeError.INCOMPATIBLE_MODEL: IncompatibleModel(),
    NativeError.COMPILATION_FAILED: CompilationFailed(),
    NativeError.INTERNAL_ERROR: InternalError(),
    NativeError.INCOMPATIBLE_API_CLIENT_ERROR: IncompatibleApiClientError(),
    NativeError.INVALID_YAML: InvalidYamlException(),
    NativeError.API_CLIENT_INIT_FAILED: ApiClientInitFailed(),
    NativeError.NO_API_KEY: NoApiKeyException(),
    NativeError.INVALID_SESSION_OPTIONS: InvalidSessionOption(),
    NativeError.QUEUE_WAIT_TIMEOUT: QueueWaitTimeout(),
    NativeError.SESSION_TERMINATED: SessionTerminated(),
    NativeError.DEVICE_BUSY: DeviceBusy(),
    NativeError.TENSOR_NAME_NOT_FOUND: TensorNameNotFound(),
    NativeError.UNSUPPORTED_FEATURE: UnsupportedFeature(),
    NativeError.INVALID_COMPILER_CONFIG: InvalidCompilerConfig(),
}


def into_exception(err: typing.Union[ctypes.c_int, int]) -> NativeException:
    """
    Convert nux_error_t type in Nux C API to NuxException

    Arguments:
        err (NativeError) NativeError converted from ``nux_error_t`` enum in C

    Returns:
        NuxException
    """
    # FIXME (@hyunsik): C APIs defined in ctypes returns c_int, or int value.
    #   There's no way to make the behavior deterministic.
    err = err.value if isinstance(err, ctypes.c_int) else err

    if err == NativeError.SUCCESS:
        return RuntimeError(msg='NuxErr.SUCCESS cannot be NuxException')

    if err in _errors_to_exceptions:
        return _errors_to_exceptions[err]

    return InternalError()
