"""A FuriosaAI qunatizer."""

import enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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
        model: Union[onnx.ModelProto, bytes],  # pylint: disable=no-member
        calibration_method: CalibrationMethod,
        *,
        percentage: float = 99.99,
    ):
        """
        Args:
            model (onnx.ModelProto or bytes): An ONNX model to
                calibrate.
            calibration_method (CalibrationMethod): A calibration
                method.
            percentage (float): A percentage to use with percentile
                calibration. Defaults to 99.99 (i.e. 99.99%-percentile
                calibration).
        """
        if isinstance(model, onnx.ModelProto):  # pylint: disable=no-member
            model = model.SerializeToString()
        self._calibrator = furiosa_quantizer_impl.Calibrator(  # pylint: disable=no-member
            model, calibration_method, percentage
        )
        self._collected_data = False

    def collect_data(self, calibration_dataset: Iterable[Sequence[np.ndarray]]) -> None:
        """Collect the values of tensors that will be used for range
        computation.

        This can be called multiple times.

        Args:
            calibration_dataset (Iterable[Sequence[numpy.ndarray]]):
                An object that provides input data for the model one at
                a time.
        """
        self._calibrator.collect_data(calibration_dataset)
        self._collected_data = True

    def _collect_data_and_return_outputs(
        self, calibration_dataset: Iterable[Sequence[np.ndarray]]
    ) -> List[List[np.ndarray]]:
        """Collect the values of tensors that will be used for range
        computation, and return outputs.

        This can be called multiple times.

        Args:
            calibration_dataset (Iterable[Sequence[numpy.ndarray]]):
                An object that provides input data for the model one at
                a time.

        Returns:
            List[List[numpy.ndarray]]: Outputs.
        """
        # pylint: disable-next=protected-access
        outputs = self._calibrator._collect_data_and_return_outputs(calibration_dataset)
        outputs = [[array.astype(typestr) for array, typestr in output] for output in outputs]
        self._collected_data = True
        return outputs

    def compute_range(self, verbose: bool = False) -> Dict[str, Tuple[float, float]]:
        """Estimate the ranges of the tensors on the basis of the collected
        data.

        Args:
            verbose (bool): Whether to show a progress bar, Defaults to
                False.

        Returns:
            Dict[str, Tuple[float, float]]: A dictionary that maps a
                tensor name to a tuple of the tensor's min and max.
        """
        if not self._collected_data:
            raise RuntimeError("collect_data must be called before compute_range")
        return self._calibrator.compute_range(verbose)


def quantize(
    model: Union[onnx.ModelProto, bytes],  # pylint: disable=no-member
    tensor_name_to_range: Mapping[str, Sequence[float]],
    *,
    with_quantize: bool = True,
    normalized_pixel_outputs: Optional[Sequence[int]] = None,
) -> Graph:
    """Quantize an ONNX model on the basis of the range of its tensors.

    Args:
        model (onnx.ModelProto or bytes): An ONNX model to quantize.
        tensor_name_to_range (Mapping[str, Sequence[float]]):
            A mapping from a tensor name to a 2-tuple (or list) of the
            tensor's min and max.
        with_quantize (bool): Whether to put a Quantize operator at the
            beginning of the resulting model. Defaults to True.
        normalized_pixel_outputs (Optional[Sequence[int]]):
            A sequence of indices of output tensors in the ONNX model
            that produce pixel values in a normalized format ranging
            from 0.0 to 1.0. If specified, the corresponding output
            tensors in the resulting quantized model will generate pixel
            values in an unnormalized format from 0 to 255, represented
            as unsigned 8-bit integers (uint8). Defaults to None.

    Returns:
        Graph: An intermediate representation (IR) of the quantized
            model.
    """
    if isinstance(model, onnx.ModelProto):  # pylint: disable=no-member
        model = model.SerializeToString()
    return furiosa_quantizer_impl.quantize(  # pylint: disable=no-member
        model, tensor_name_to_range, with_quantize, normalized_pixel_outputs
    )
