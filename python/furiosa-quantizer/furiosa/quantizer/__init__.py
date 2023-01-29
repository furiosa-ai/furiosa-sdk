"""A FuriosaAI qunatizer."""

import enum
from typing import Iterable, Mapping, Sequence, Union

import furiosa_quantizer_impl
from furiosa_quantizer_impl import Graph  # pylint: disable=no-name-in-module
import numpy as np
import onnx

import furiosa.common.utils

__version__ = furiosa.common.utils.get_sdk_version(__name__)

__all__ = ["CalibrationMethod", "Calibrator", "Graph", "quantize"]


CalibrationMethod = enum.IntEnum(
    "CalibrationMethod",
    # pylint: disable=no-member
    [
        (name, getattr(furiosa_quantizer_impl.CalibrationMethod, name))
        for name in dir(furiosa_quantizer_impl.CalibrationMethod)
        if not name.startswith("_")
    ],
    module=__name__,
    qualname="CalibrationMethod",
)
CalibrationMethod.__doc__ = (
    furiosa_quantizer_impl.CalibrationMethod.__doc__  # pylint: disable=no-member
)


class Calibrator:
    """Calibrator.

    This collects the values of tensors in an ONNX model and computes
    their ranges.
    """

    def __init__(
        self,
        model: Union[onnx.ModelProto, bytes],
        calibration_method: CalibrationMethod,
        percentage=99.99,
    ):
        """
        Args:
            model (onnx.ModelProto or bytes): An ONNX model to
                calibrate.
            calibration_method (CalibrationMethod): A calibration
                method.
            percentage (float, optional): A percentage to use with
                percentile calibration. Defaults to 99.99 (i.e.
                99.99%-percentile calibration).
        """
        if isinstance(model, onnx.ModelProto):
            model = model.SerializeToString()
        self._calibrator = furiosa_quantizer_impl.Calibrator(  # pylint: disable=no-member
            model, calibration_method, percentage
        )

    def collect_data(self, calibration_dataset: Iterable[Sequence[np.ndarray]]):
        """Collect the values of tensors that will be used for range
        computation.

        This can be called multiple times.

        Args:
            calibration_dataset (Iterable[Sequence[numpy.ndarray]]):
                An object that provides input data for the model one at
                a time.
        """
        return self._calibrator.collect_data(calibration_dataset)

    def compute_range(self, verbose=False):
        """Estimate the ranges of the tensors on the basis of the collected
        data.

        Args:
            verbose (bool): Whether to show a progress bar, Defaults to
                False.
        """
        return self._calibrator.compute_range(verbose)


def quantize(
    model: Union[onnx.ModelProto, bytes], tensor_name_to_range: Mapping[str, Sequence[float]]
) -> Graph:
    """Quantize an ONNX model on the basis of the range of its tensors.

    Args:
        model (onnx.ModelProtoo or bytes): An ONNX model to quantize.
        tensor_name_to_range (Mapping[str, Sequence[float]]):
            A mapping from a tensor name to a 2-tuple (or list) of the
            tensor's min and max.

    Returns:
        Graph: An intermediate representation (IR) of the quantized
            model.
    """
    if isinstance(model, onnx.ModelProto):
        model = model.SerializeToString()
    return furiosa_quantizer_impl.quantize(model, tensor_name_to_range)  # pylint: disable=no-member
