"""Session and its asynchronous API for model inference"""

import ctypes
from ctypes import byref, c_int, c_void_p
from pathlib import Path
import typing
from typing import Dict, List, Optional, Union

import numpy as np
import yaml

from . import envs
from ._api import LIBNUX
from ._api.v1 import convert_to_cchar_array, decref, increase_ref_count, runtime_version
from ._util import dump_info
from .compiler import _model_image, generate_compiler_log_path
from .errors import (
    InvalidInput,
    SessionClosed,
    UnsupportedTensorType,
    into_exception,
    is_err,
    is_ok,
)
from .model import Model, TensorArray
from .tensor import Tensor, TensorDesc


def _fill_tensor(value: Union[np.ndarray, np.generic], target: Tensor):
    if value.shape != target.shape or value.dtype != target.numpy_dtype:
        raise InvalidInput(
            f"expected {target.shape} ({target.numpy_dtype}) but got {value.shape} ({value.dtype})"
        )

    target.copy_from(value)


def _fill_all_tensors(
    values: Union[np.ndarray, np.generic, TensorArray, List[Union[np.ndarray, np.generic]]],
    targets: TensorArray,
) -> TensorArray:
    """
    Fills `targets` with buffers copied from `values`
    """
    if isinstance(values, (np.ndarray, np.generic)):
        _fill_tensor(values, targets[0])
        return targets

    if isinstance(values, list):
        if len(values) != targets.len:
            raise InvalidInput(
                f"{targets.len} tensors are expected, " f"but {len(values)} tensors are given"
            )

        for value, target in zip(values, targets):
            _fill_tensor(value, target)

        return targets

    if isinstance(values, TensorArray):
        return values

    raise UnsupportedTensorType()


def _create_session_options(
    device: Optional[str] = None,
    worker_num: Optional[int] = None,
    compile_config: Optional[Dict[str, object]] = None,
    compiler_log: Optional[Path] = None,
    input_queue_size: Optional[int] = None,
    output_queue_size: Optional[int] = None,
):
    options: c_void_p = LIBNUX.nux_session_option_create()
    if device:
        LIBNUX.nux_session_option_set_device(options, device.encode())
    if worker_num:
        LIBNUX.nux_session_option_set_worker_num(options, worker_num)
    if compile_config:
        compile_config = yaml.dump(compile_config).encode()
        LIBNUX.nux_session_option_set_compiler_config(options, compile_config)
    if compiler_log:
        compiler_log = str(compiler_log).encode()
        LIBNUX.nux_session_option_set_compiler_log_path(options, compiler_log)
    if input_queue_size:
        LIBNUX.nux_session_option_set_input_queue_size(options, input_queue_size)
    if output_queue_size:
        LIBNUX.nux_session_option_set_output_queue_size(options, output_queue_size)

    return options


class Session(Model):
    """Provides a blocking API to run an inference task with a given model"""

    def __init__(
        self,
        model: Union[bytes, str, Path],
        device: str = None,
        worker_num: int = None,
        compile_config: Dict[str, object] = None,
    ):

        if device is None:
            device = envs.current_npu_device()
        log_path = generate_compiler_log_path()
        options = _create_session_options(device, worker_num, compile_config, log_path)
        model_image = _model_image(model)

        print(f"Using furiosa-compiler {runtime_version()}")
        sess = c_void_p(None)
        err = LIBNUX.nux_session_create(model_image, len(model_image), options, byref(sess))
        if is_err(err):
            dump_info(log_path)
            raise into_exception(err)

        self.ref = sess
        self._as_parameter_ = self.ref

        super().__init__()

    def _get_model_ref(self) -> c_void_p:
        if self.ref:
            return LIBNUX.nux_session_get_model(self)
        else:
            raise SessionClosed()

    def run(
        self,
        inputs: Union[np.ndarray, np.generic, TensorArray, List[Union[np.ndarray, np.generic]]],
    ) -> TensorArray:
        """
        Runs an inference task with `inputs`

        Args:
            inputs: It can be a single runtime.Tensor, runtime.TensorArray or \
            numpy.ndarray object. Also, you can pass one TensorArray or a \
            list of numpy.ndarray objects.

        Returns:
            Inference output
        """
        if not self.ref:
            raise SessionClosed

        _inputs = self.allocate_inputs()
        outputs = self.create_outputs()
        _inputs = _fill_all_tensors(inputs, _inputs)

        err = LIBNUX.nux_session_run(self.ref, _inputs, outputs)

        if is_err(err):
            raise into_exception(err)

        return outputs

    def run_with(
        self, outputs: typing.List[str], inputs: Dict[str, Union[np.ndarray]]
    ) -> TensorArray:
        """
        Runs an inference task with `inputs`

        Args:
            inputs: It can be a single runtime.Tensor, runtime.TensorArray or \
            numpy.ndarray object. Also, you can pass one TensorArray or a \
            list of numpy.ndarray objects.

        Returns:
            Inference output
        """
        if not self.ref:
            raise SessionClosed

        # FIXME: outputs=None should be supported in the future
        if outputs is None:
            raise InvalidInput(message="outputs must be given")

        input_names = [name for name in inputs.keys()]
        input_tensors = self.allocate_tensors(input_names)
        output_tensors = self.create_tensors(outputs)
        input_tensors = _fill_all_tensors([value for value in inputs.values()], input_tensors)

        input_names_ptr = convert_to_cchar_array(input_names)
        output_names_ptr = convert_to_cchar_array(outputs)

        err = LIBNUX.nux_session_run_with(
            self.ref,
            input_names_ptr,
            len(input_names),
            output_names_ptr,
            len(outputs),
            input_tensors,
            output_tensors,
        )
        if is_err(err):
            raise into_exception(err)

        return output_tensors

    def close(self):
        """Close the session and release all resources belonging to the session"""
        if self.ref:
            LIBNUX.nux_session_destroy(self.ref)
            self.ref = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


class CompletionQueue:
    """Receives the completion results asynchronously from AsyncSession"""

    def __init__(self, ref: c_void_p, context_ty: type, output_descs: [TensorDesc]):
        self._as_parameter_ = ref
        self.ref = ref
        self.context_ty = context_ty
        self.output_descs = output_descs
        self.queue_ok = True

    def recv(self, timeout: Optional[int] = None) -> (object, TensorArray):
        """Receives the prediction results which are asynchronously coming from AsyncSession

        If there are already prediction outputs, it will return immediately.
        Otherwise, it will be blocked until the next result are ready.

        If ``timeout`` is set, ``recv()`` will be blocked only until
        the timeout occurs. If timed out, ``recv()`` throws ``QueueWaitTimeout``
        exception.

        If AsyncSession is closed earlier ``recv()`` will throw
        ``SessionTerminated`` exception.

        Args:
            timeout (int): How long to wait before giving up.
            It should be a positive interger in milliseconds.

        Returns:
            A tuple, whose first value is the context value passed \
            when you submit an inference task and the second value \
            is inference output.
        """
        if not self.ref:
            raise SessionClosed()

        err = c_int(0)
        context_ref = ctypes.py_object(None)
        outputs_ref = c_void_p(None)

        if timeout:
            if timeout < 0:
                raise RuntimeError("the timeout duration must be a positive integer")
            self.queue_ok = LIBNUX.nux_completion_queue_next_timeout(
                self.ref, timeout, byref(context_ref), byref(outputs_ref), byref(err)
            )
        else:
            self.queue_ok = LIBNUX.nux_completion_queue_next(
                self.ref, byref(context_ref), byref(outputs_ref), byref(err)
            )
        context_val = context_ref.value
        decref(context_ref)

        if is_ok(err):
            return context_val, TensorArray(outputs_ref, self.output_descs, allocated=False)

        raise into_exception(err)

    def close(self):
        """Closes this completion queue.

        If it is closed, AsyncSession also will stop working.
        """
        if self.ref:
            LIBNUX.nux_completion_queue_destroy(self.ref)
            self.ref = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.queue_ok:
            return self.recv()

        raise StopIteration()


class AsyncSession(Model):
    """An asynchronous session for a given model allows to submit predictions"""

    def __init__(self, ref: c_void_p):
        self._ref = ref
        self._as_parameter_ = self._ref
        super().__init__()

        self._tensor_buffers = self.allocate_inputs()

    def _get_model_ref(self) -> c_void_p:
        if self._ref:
            return LIBNUX.nux_async_session_get_model(self)
        else:
            raise SessionClosed()

    def submit(
        self, values: Union[np.ndarray, np.generic, TensorArray], context: object = None
    ) -> None:
        """
        Submit a prediction request

        It immediately returns without blocking the caller, and
        If the prediction is completed, the outputs will be sent to CompletionQueue.

        Args:
            values: Input values
            context: an additional context to identify the prediction request
        """
        _fill_all_tensors(values, self._tensor_buffers)
        # manually increase reference count to keep the context object while running
        increase_ref_count(context)
        err = LIBNUX.nux_async_session_run(self._ref, context, self._tensor_buffers)

        if is_err(err):
            raise into_exception(err)

    def close(self):
        """Closes this session

        After a session is closed, CompletionQueue will return an error
        if CompletionQueue.recv() is called.
        """
        if self._ref:
            LIBNUX.nux_async_session_destroy(self._ref)
            self._ref = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def create(
    model: Union[bytes, str, Path],
    device: str = None,
    worker_num: int = None,
    compile_config: Dict[str, object] = None,
) -> Session:
    """Creates a session for a model

    Args:
        model (bytes, str, Path): a byte string containing a model image or \
        a path string of a model image file
        device: NPU device (str) (e.g., npu0pe0, npu0pe0-1)
        worker_num: Number of workers
        compile_config (Dict[str, object]): Compile config

    Returns:
        the session for a given model, allowing to run predictions. \
        Session is a thread safe.
    """
    return Session(model, device, worker_num, compile_config)


def create_async(
    model: Union[bytes, str, Path],
    context_ty: type = None,
    device: str = None,
    worker_num: int = None,
    input_queue_size: int = None,
    output_queue_size: int = None,
    compile_config: Dict[str, object] = None,
) -> (AsyncSession, CompletionQueue):
    """Creates a pair of the asynchronous session and the completion queue for a given model

    Args:
        model (bytes, str, Path): a byte string containing a model image or \
        a path string of a model image file
        context_ty (type): Type for passing context from AsyncSession to CompletionQueue
        device: NPU device (str) (e.g., npu0pe0, npu0pe0-1)
        worker_num: Number of workers
        input_queue_size: The input queue size, and it must be > 0 and < 2^31.
        output_queue_size: The output queue size, and it must be be > 0 and < 2^31.
        compile_config (Dict[str, object]): Compile config

    Returns:
        A pair of the asynchronous session and the completion queue. \
        the asynchronous session for a given model allows to submit predictions. \
        the completion queue allows users to receive the prediction outputs \
        asynchronously.
    """

    try:
        if device is None:
            device = envs.current_npu_device()

        model_image = _model_image(model)
        log_path = generate_compiler_log_path()
        options = _create_session_options(
            device, worker_num, compile_config, log_path, input_queue_size, output_queue_size
        )

        print(f"Using furiosa-compiler {runtime_version()}")
        sess_ref = c_void_p(None)
        queue_ref = c_void_p(None)
        err = LIBNUX.nux_async_session_create(
            model_image, len(model_image), options, byref(sess_ref), byref(queue_ref)
        )
        if is_ok(err):
            sess = AsyncSession(sess_ref)
            return sess, CompletionQueue(queue_ref, context_ty, sess.outputs())
        else:
            dump_info(log_path)
            raise into_exception(err)
    finally:
        pass
