"""Session and its asynchronous API for model inference"""

import ctypes
from ctypes import byref, c_int32, c_void_p
from typing import Union

import numpy as np

from ._api import LIBNUX
from ._api.v1 import decref, increase_ref_count
from .errors import UnsupportedTensorType, into_exception, is_err, is_ok
from .model import Model, TensorArray
from .tensor import TensorDesc


def _fill_tensors(values: Union[np.ndarray, np.generic, TensorArray],
                  targets: TensorArray) -> TensorArray:
    """
    Fills `targets` with buffers copied from `values`
    """
    if isinstance(values, (np.ndarray, np.generic)):
        targets[0].copy_from(values)
        return targets

    if isinstance(values, list):
        for idx, value in enumerate(values):
            targets[idx].copy_from(value)
        return targets

    if isinstance(values, TensorArray):
        return values

    raise UnsupportedTensorType()


class Session(Model):
    """Provides a blocking API to run an inference task with a given model"""
    ref = c_void_p(None)

    def __init__(self, model):
        sess = c_void_p(None)
        options: c_void_p = LIBNUX.nux_session_option_create()

        model_image = _model_image(model)

        err = LIBNUX.nux_session_create(model_image, len(model_image), options, byref(sess))
        if is_err(err):
            raise into_exception(err)

        self.ref = sess
        self._as_parameter_ = self.ref

        super().__init__()

    def _get_model_ref(self) -> c_void_p:
        return LIBNUX.nux_session_get_model(self)

    def run(self, inputs) -> TensorArray:
        """
        Runs an inference task with `inputs`

        Args:
            inputs: It can be a single runtime.Tensor, runtime.TensorArray or \
            numpy.ndarray object. Also, you can pass one TensorArray or a \
            list of numpy.ndarray objects.

        Returns:
            Inference output
        """
        _inputs = self.allocate_inputs()
        outputs = self.create_outputs()
        _inputs = _fill_tensors(inputs, _inputs)

        err = LIBNUX.nux_session_run(self.ref, _inputs, outputs)

        if is_err(err):
            raise into_exception(err)

        return outputs

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
    ref = c_void_p(None)
    context_ty: type
    output_descs: [TensorDesc]

    def __init__(self, ref: c_void_p, context_ty: type, output_descs: [TensorDesc]):
        self._as_parameter_ = ref
        self.ref = ref
        self.context_ty = context_ty
        self.output_descs = output_descs
        self.queue_ok = True

    def recv(self, timeout: int = None) -> (object, TensorArray):
        """Receives the prediction results asynchronously coming from AsyncSession

        If there are already prediction outputs, it will return immediately.
        Or it will be blocked until the next result are ready.

        If ``timeout`` is set, ``recv()`` will be only blocked until timeout occurs.
        Then, ``recv()`` throws ``SessionTerminated`` exception.

        Args:
            timeout (int): How long to wait before giving up.
            It should be a positive interger in milliseconds.

        Returns:
            A tuple, whose first value is the context value passed \
            when you submit an inference task and the second value \
            is inference output.
        """
        err = c_int32(0)
        context_ref = ctypes.py_object(None)
        outputs_ref = c_void_p(None)

        if timeout:
            if timeout < 0:
                raise RuntimeError("the timeout duration must be a positive integer")
            self.queue_ok = LIBNUX.nux_completion_queue_next_timeout(self.ref,
                                                                     timeout,
                                                                     byref(context_ref),
                                                                     byref(outputs_ref),
                                                                     byref(err))
        else:
            self.queue_ok = LIBNUX.nux_completion_queue_next(self.ref,
                                                             byref(context_ref),
                                                             byref(outputs_ref),
                                                             byref(err))
        context_val = context_ref.value
        decref(context_ref)

        if is_ok(err.value):
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
    ref = c_void_p(None)
    inputs: TensorArray

    def __init__(self, ref: c_void_p):
        self.ref = ref
        self._as_parameter_ = self.ref
        super().__init__()

        self.inputs = self.allocate_inputs()

    def _get_model_ref(self) -> c_void_p:
        return LIBNUX.nux_async_session_get_model(self)

    def submit(self, values: Union[np.ndarray, np.generic, TensorArray],
               context: object = None) -> None:
        """
        Submit a prediction request

        It immediately returns without blocking the caller, and
        If the prediction is completed, the outputs will be sent to CompletionQueue.

        Args:
            values: Input values
            context: an additional context to identify the prediction request
        """
        _fill_tensors(values, self.inputs)
        # manually increase reference count to keep the context object while running
        increase_ref_count(context)
        err = LIBNUX.nux_async_session_run(self.ref, context, self.inputs)

        if is_err(err):
            raise into_exception(err)

    def close(self):
        """Closes this session

        After a session is closed, CompletionQueue will return an error
        if CompletionQueue.recv() is called.
        """
        if self.ref:
            LIBNUX.nux_async_session_destroy(self.ref)
            self.ref = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def _read_file(path):
    with open(path, 'rb') as file:
        contents = file.read()
        return contents


def _model_image(model) -> bytes:
    if isinstance(model, bytes):
        model_image = model
    elif isinstance(model, str):
        model_image = _read_file(model)
    else:
        raise TypeError("'model' must be str or bytes, but it was " + repr(type(model)))

    return model_image


def create(model) -> Session:
    """Creates a session for a model

    Args:
        model (bytes or str): a byte string containing a model image or \
        a path string of a model image file

    Returns:
        the session for a given model, allowing to run predictions. \
        Session is a thread safe.
    """
    return Session(model)


def create_async(model, context_ty: type = None) -> (AsyncSession, CompletionQueue):
    """Creates a pair of the asynchronous session and the completion queue for a given model

    Args:
        model (bytes or str): a byte string containing a model image or \
        a path string of a model image file

    Returns:
        A pair of the asynchronous session and the completion queue. \
        the asynchronous session for a given model allows to submit predictions. \
        the completion queue allows users to receive the prediction outputs \
        asynchronously.
    """

    try:
        model_image = _model_image(model)

        options: c_void_p = LIBNUX.nux_session_option_create()
        sess_ref = c_void_p(None)
        queue_ref = c_void_p(None)
        err = LIBNUX.nux_async_session_create(model_image, len(model_image), options,
                                              byref(sess_ref), byref(queue_ref))
        if is_ok(err):
            sess = AsyncSession(sess_ref)
            return sess, CompletionQueue(queue_ref, context_ty, sess.outputs())

        raise into_exception(err)
    finally:
        pass
