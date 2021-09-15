"""Nux Exception and Error"""
import ctypes
import sys
import typing
from enum import IntEnum
from typing import Optional


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


def is_ok(err: NativeError) -> bool:
    """True if NuxErr is SUCCESS, or False"""
    return err == NativeError.SUCCESS


def is_err(self) -> bool:
    """True if NuxErr is not SUCCESS, or False"""
    return self != NativeError.SUCCESS


class NuxException(Exception):
    """general exception caused by Nuxpy"""
    native_err: Optional[NativeError]
    msg: str

    def __init__(self, msg: str, native_err: NativeError = None):
        self.native_err = native_err
        self.msg = msg
        super().__init__(self)

    def message(self) -> str:
        """Error message"""
        return self.msg

    def native_error(self) -> Optional[NativeError]:
        """Return a native error if this exception comes from C native extension"""
        return self.native_err

    def __repr__(self):
        if self.native_err is None:
            return '{}'.format(self.msg)

        return '{} (native error code: {})'.format(self.msg, self.native_err)

    def __str__(self):
        return self.__repr__()


class IncompatibleModel(NuxException):
    """When Renegade compiler cannot recognize a given model image binary"""

    def __init__(self):
        super().__init__("Model binary is not compatible",
                         NativeError.INCOMPATIBLE_MODEL)


class CompilationFailed(NuxException):
    """when Nux fails to compile a given model image to NPU model binary"""

    def __init__(self):
        super().__init__("fail to compile a given model to NPU binary",
                         NativeError.COMPILATION_FAILED)


class InternalError(NuxException):
    """internal error or no corresponding error in Python binding"""

    def __init__(self, cause='unknown'):
        super().__init__("{}".format(cause), NativeError.INTERNAL_ERROR)


class UnsupportedTensorType(NuxException):
    """Unsupported tensor type"""

    def __init__(self):
        super().__init__("numpy.ndarray, TensorArray are only supported",
                         NativeError.INVALID_INPUTS)


class UnsupportedDataType(NuxException):
    """Unsupported tensor data type"""

    def __init__(self, dtype):
        super().__init__(msg="unknown data type: {}".format(dtype))


class IncompatibleApiClientError(NuxException):
    """When both API client and server are incompatible"""

    def __init__(self):
        super().__init__("incompatible client with Furiosa API server",
                         NativeError.INCOMPATIBLE_API_CLIENT_ERROR)


class InvalidYamlException(NuxException):
    """When Renegade compiler cannot recognize a given model image binary"""

    def __init__(self):
        super().__init__("Compiler config is not valid YAML",
                         NativeError.INVALID_YAML)


class ApiClientInitFailed(NuxException):
    """when api client fails to initialize due to api keys or others"""

    def __init__(self):
        super().__init__("fail to initialize API client",
                         NativeError.API_CLIENT_INIT_FAILED)


class NoApiKeyException(NuxException):
    """when api client fails to initialize due to api keys or others"""

    def __init__(self):
        super().__init__("No API keys. Please check your API keys.",
                         NativeError.NO_API_KEY)


class InvalidSessionOption(NuxException):
    """when api client fails to initialize due to api keys or others"""

    def __init__(self):
        super().__init__("Invalid options passed to session.create() or create_async()",
                         NativeError.INVALID_SESSION_OPTIONS)


class QueueWaitTimeout(NuxException):
    """when api client fails to initialize due to api keys or others"""

    def __init__(self):
        super().__init__("Queue waiting timed out",
                         NativeError.QUEUE_WAIT_TIMEOUT)


class SessionTerminated(NuxException):
    """when api client fails to initialize due to api keys or others"""

    def __init__(self):
        super().__init__("Session or AsyncSession terminated",
                         NativeError.SESSION_TERMINATED)


class DeviceBusy(NuxException):
    """when api client fails to initialize due to api keys or others"""

    def __init__(self):
        super().__init__("NPU device busy",
                         NativeError.DEVICE_BUSY)


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
}

def into_exception(err: typing.Union[ctypes.c_int, int]) -> NuxException:
    """
    Convert nux_error_t type in Nux C API to NuxException

    Arguments:
        err (NativeError) NativeError converted from ``nux_error_t`` enum in C

    Returns:
        NuxException
    """
    # FIXME (@hyunsik): C APIs defined in ctypes returns c_int, or int value.
    #   There's no way to make the behavior deterministic.
    if isinstance(err, ctypes.c_int):
        err = err.value
    elif isinstance(err, int):
        pass

    if err == NativeError.SUCCESS:
        return RuntimeError(msg='NuxErr.SUCCESS cannot be NuxException')

    if err in _errors_to_exceptions.keys():
        return _errors_to_exceptions[err]

    return InternalError()
