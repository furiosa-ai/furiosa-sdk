"""A FuriosaAI qunatizer."""

from typing import Mapping, Sequence, Union

import furiosa_quantizer_impl
import onnx

import furiosa.common.utils
from furiosa.quantizer.calibrator import CalibrationMethod, Calibrator
from furiosa.quantizer.editor import ModelEditor, TensorType, get_output_names, get_pure_input_names

__version__ = furiosa.common.utils.get_sdk_version(__name__)

__full_version__ = f"Furiosa SDK Quantizer {__version__} (furiosa_quantizer_impl {furiosa_quantizer_impl.__version__} {furiosa_quantizer_impl.__git_short_hash__} {furiosa_quantizer_impl.__build_timestamp__})"  # pylint: disable=no-member # noqa: E501

__all__ = [
    "CalibrationMethod",
    "Calibrator",
    "get_pure_input_names",
    "get_output_names",
    "ModelEditor",
    "TensorType",
    "quantize",
]


def quantize(
    model: Union[onnx.ModelProto, bytes],  # pylint: disable=no-member
    tensor_name_to_range: Mapping[str, Sequence[float]],
) -> bytes:
    """Quantize an ONNX model on the basis of the range of its tensors.

    Args:
        model (onnx.ModelProto or bytes): An ONNX model to quantize.
        tensor_name_to_range (Mapping[str, Sequence[float]]):
            A mapping from a tensor name to a 2-tuple (or list) of the
            tensor's min and max.

    Returns:
        bytes: A serialized ONNX model that incorporates quantization
            information.
    """
    if isinstance(model, onnx.ModelProto):  # pylint: disable=no-member
        model = model.SerializeToString()
    return furiosa_quantizer_impl.quantize(model, tensor_name_to_range)  # pylint: disable=no-member
