"""Tensor object and its utilities"""

import ctypes
from ctypes import POINTER, byref, c_uint8, c_uint64, c_void_p
from enum import IntEnum
from typing import Optional, Tuple, Union

import numpy as np

from ._api import LIBNUX
from ._util import list_to_dict
from .errors import UnsupportedTensorType, is_err


class Axis(IntEnum):
    """Axis of Tensor"""

    WIDTH = 0
    HEIGHT = 1
    CHANNEL = 2
    BATCH = 3
    WIDTH_OUTER = 4
    HEIGHT_OUTER = 5
    CHANNEL_OUTER = 6
    BATCH_OUTER = 7
    UNKNOWN = 8

    @classmethod
    def _names(cls):
        return ["W", "H", "C", "N", "Wo", "Ho", "Co", "No", "?"]

    def __repr__(self):
        return self._names()[self]


class DataType(IntEnum):
    """Tensor data type"""

    FLOAT32 = 0
    UINT8 = 1
    INT8 = 2
    INT32 = 3
    INT64 = 4
    BFLOAT16 = 5  # Not supported yet in Numpy

    @classmethod
    def _numpy_dtypes(cls):
        return [np.float32, np.uint8, np.int8, np.int32, np.int64]

    def __repr__(self) -> str:
        return self._name_

    @property
    def numpy_dtype(self):
        """Return the numpy dtype corresponding to this DataType"""
        return self._numpy_dtypes()[self]


class TensorDesc:
    """Tensor description including dimension, shape, and data type"""

    def __init__(self, ref: c_void_p):
        self.ref = ref
        self._as_parameter_ = ref

    @property
    def name(self) -> Optional[str]:
        name_ptr = LIBNUX.nux_tensor_name(self)
        if name_ptr:
            name = ctypes.c_char_p(name_ptr).value.decode("utf-8")
            LIBNUX.nux_string_destroy(name_ptr)
            return name
        else:
            return None

    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return LIBNUX.nux_tensor_dim_num(self)

    def dim(self, idx: int) -> int:
        """Size of i-th dimension"""
        return LIBNUX.nux_tensor_dim(self, idx)

    @property
    def shape(self) -> tuple:
        """tensor shape"""
        dims = []
        for i in range(self.ndim):
            dims.append(self.dim(i))

        return tuple(dims)

    def axis(self, idx: int) -> Axis:
        """Axis type of i-th dimension (e.g., width, height, channel)"""
        return Axis(LIBNUX.nux_tensor_axis(self, idx))

    @property
    def size(self) -> int:
        """Size in bytes"""
        return LIBNUX.nux_tensor_size(self)

    def stride(self, idx: int) -> int:
        """Stride of i-th dimension"""
        return LIBNUX.nux_tensor_stride(self, idx)

    @property
    def length(self) -> int:
        """Number of all elements across all dimensions"""
        return LIBNUX.nux_tensor_len(self)

    @property
    def format(self) -> str:
        """Tensor memory layout (e.g., NHWC, NCHW)"""
        tensor_format = str()
        for idx in range(self.ndim):
            tensor_format += self.axis(idx).__repr__()

        return tensor_format

    @property
    def dtype(self) -> DataType:
        """Data type of tensor"""
        return DataType(LIBNUX.nux_tensor_dtype(self))

    @property
    def numpy_dtype(self):
        """Return numpy dtype"""
        return self.dtype.numpy_dtype

    @property
    def quantization_parameter(self) -> Tuple[ctypes.c_double, ctypes.c_int32]:
        """Return the quantization parameter for this tensor"""
        scale = ctypes.c_double(0.0)
        zero_point = ctypes.c_int32(0)

        err = LIBNUX.nux_tensor_quantization_parameter(self, byref(scale), byref(zero_point))
        if is_err(err):
            raise into_exception(err)
        return (scale, zero_point)

    def __repr__(self) -> str:
        repr = self.__class__.__name__ + "("
        if self.name:
            repr += f"name=\"{self.name}\", "

        repr += (
            f"shape={self.shape}, dtype={self.dtype.__repr__()}, "
            f"format={self.format}, size={self.size}, len={self.length})"
        )

        return repr


class Tensor:
    """A tensor which contains data and tensor description including shape"""

    def __init__(self, ref: c_void_p, desc: TensorDesc, allocated: bool = False):
        self.ref = ref
        self.desc = desc
        self.allocated = allocated
        self._as_parameter_ = ref

    @property
    def shape(self) -> tuple:
        """Return the tensor shape

        Returns:
            Tensor shape. An example shape is
            ```(1, 28, 28, 1)```.
        """
        return self.desc.shape

    @property
    def dtype(self) -> DataType:
        """Data type of tensor"""
        return self.desc.dtype

    @property
    def numpy_dtype(self):
        """Return numpy dtype"""
        return self.desc.numpy_dtype

    @property
    def quantization_parameter(self) -> Tuple[ctypes.c_double, ctypes.c_int32]:
        """Return the quantization parameter for this tensor"""

        return self.desc.quantization_parameter

    def copy_from(self, data: Union[np.ndarray, np.generic]):
        """Copy the contents of Numpy ndarray to this tensor"""
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data)
        if isinstance(data, (np.ndarray, np.generic)):
            buf_ptr = data.ctypes.data_as(POINTER(c_uint8))
            buf_size_in_bytes = data.nbytes
            LIBNUX.tensor_fill_buffer(self, buf_ptr, buf_size_in_bytes)
        else:
            raise UnsupportedTensorType()

    def view(self) -> np.ndarray:
        """Return numpy.ndarray view converted from this tensor"""
        arr_size = self.desc.size

        buf_ptr = POINTER(c_uint8)()
        buf_len = c_uint64(0)
        LIBNUX.tensor_get_buffer(self, byref(buf_ptr), byref(buf_len))

        buf_from_mem = ctypes.pythonapi.PyMemoryView_FromMemory
        buf_from_mem.restype = ctypes.py_object
        buf_from_mem.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
        buffer = buf_from_mem(buf_ptr, arr_size, 0x100)

        itemsize = np.dtype(self.numpy_dtype).itemsize
        strides = []
        for i in range(self.desc.ndim):
            strides.append(self.desc.stride(i) * itemsize)
        strides = tuple(strides)

        arr = np.ndarray(tuple(self.shape[:]), self.numpy_dtype, buffer, strides=strides)
        return arr

    def numpy(self) -> np.ndarray:
        """Return numpy.ndarray converted from this tensor"""
        return self.view().copy()

    def __repr__(self):
        repr = self.__class__.__name__ + "("

        if self.desc.name:
            repr += f'name="{self.desc.name}", '

        repr += f"shape={self.desc.shape}, dtype={self.desc.dtype.__repr__()})"

        return repr

    # _LIBNUX is meant to keep a valid reference to the Nux library at
    # interpreter shutdown. From https://bugs.python.org/issue5099#msg80855:
    #
    # > At interpreter shutdown, the module's global variables are set to None
    # > before the module itself is released. __del__ methods may be called in
    # > those precarious circumstances, and should not rely on any global
    # > state.
    def _close(self, _LIBNUX=LIBNUX):
        if self.allocated and self.ref:
            _LIBNUX.nux_tensor_destroy(self.ref)
            self.allocated = False

    def __del__(self):
        self._close()

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return np.array_equal(self.numpy(), other.numpy())

        return False


class TensorArray:
    """A list of tensors

    It is used for input and output values of model inferences.
    """

    def __init__(self, ref: c_void_p, descs: [TensorDesc], allocated: bool = False):
        self.ref = ref
        self.descs = descs
        self.should_drop = allocated
        self._as_parameter_ = ref
        self.len = LIBNUX.nux_tensor_array_len(self.ref)

    def is_empty(self) -> bool:
        """True if it has no Tensor"""
        return self.len == 0

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        if isinstance(key, int):
            while key < 0:
                # Index is a negative, so addition will subtract.
                key += len(self)

            if key >= len(self):
                raise IndexError("tensor index (%d) out of range [0, %d)" % (key, len(self)))

            return Tensor(
                LIBNUX.nux_tensor_array_get(self, key), desc=self.descs[key], allocated=False
            )

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        raise TypeError

    def __setitem__(self, key, data):
        if isinstance(key, int):
            while key < 0:
                # Index is a negative, so addition will subtract.
                key += len(self)

            if key >= len(self):
                raise IndexError("tensor index (%d) out of range [0, %d)" % (key, len(self)))

            self[key].copy_from(data)
            return

        raise TypeError

    def view(self) -> [np.ndarray]:
        """Convert TensorArray to a list of numpy.ndarray view"""
        array = []
        for idx in range(self.len):
            array.append(self[idx].view())

        return array

    def numpy(self) -> [np.ndarray]:
        """Convert TensorArray to a list of numpy.ndarray"""
        array = []
        for idx in range(self.len):
            array.append(self[idx].numpy())

        return array

    # _LIBNUX is meant to keep a valid reference to the Nux library at
    # interpreter shutdown. From https://bugs.python.org/issue5099#msg80855:
    #
    # > At interpreter shutdown, the module's global variables are set to None
    # > before the module itself is released. __del__ methods may be called in
    # > those precarious circumstances, and should not rely on any global
    # > state.
    def _close(self, _LIBNUX=LIBNUX):
        if self.should_drop and self.ref:
            _LIBNUX.nux_tensor_array_destroy(self.ref)
            self.should_drop = False

    def __del__(self):
        self._close()

    def __repr__(self):
        return list_to_dict(self).__repr__()


def numpy_dtype(value):
    """Return numpy dtype from any eligible object of Nux"""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.dtype
    if isinstance(value, Tensor):
        return value.numpy_dtype
    if isinstance(value, TensorDesc):
        return value.numpy_dtype
    if isinstance(value, DataType):
        return value.numpy_dtype

    raise TypeError


def rand(tensor: TensorDesc) -> np.ndarray:
    """Return a new array of given shape and type, filled with random numbers."""
    return np.random.rand(*tensor.shape).astype(tensor.numpy_dtype)


def zeros(tensor: TensorDesc) -> np.ndarray:
    """Return a new array of given shape and type, filled with zeros."""
    return np.zeros(shape=tensor.shape, dtype=tensor.numpy_dtype)
