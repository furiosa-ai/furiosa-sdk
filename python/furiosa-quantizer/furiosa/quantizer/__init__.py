"""A FuriosaAI qunatizer."""

import enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import furiosa_quantizer_impl
import numpy as np
import onnx

import furiosa.common.utils

__version__ = furiosa.common.utils.get_sdk_version(__name__)

__full_version__ = f"Furiosa SDK Quantizer {__version__} (furiosa_quantizer_impl {furiosa_quantizer_impl.__version__} {furiosa_quantizer_impl.__git_short_hash__} {furiosa_quantizer_impl.__build_timestamp__})"  # pylint: disable=no-member

__all__ = ["CalibrationMethod", "Calibrator", "Processor", "quantize"]


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


class Processor:
    """Processor.

    Processor, which handles pre/post process functions
    and additional optimizations at model input and output.
    """

    def __init__(
        self,
        model: onnx.ModelProto,
    ):
        """
        Args:
            model (onnx.ModelProto): An ONNX model to
                be applied.
        """
        if not isinstance(model, onnx.ModelProto):
            raise RuntimeError("model type is not onnx.ModelProto")
        self._processor = furiosa_quantizer_impl.Processor(model)

    def use_u8_for_input(
        self,
        input_name: str,
    ):
        """
        Use model input as u8 type.
        This optimization is more efficient to our hardware
        and it can process image pixel data directly.

        Args:
            input_name (str): Input tensor name to apply optimization
        """
        self._processor.use_u8_for_input(input_name)

    def use_u8_for_output(self, output_name: str, tensor_range: Optional[Iterable[float]] = None):
        """
        Use model output as u8 type.
        You can directly receive the computational results of the model
        as a u8 type without converting them to f32.

        Args:
        output_name (str): Output tensor name to apply optimization
        tensor_range (Optional[Iterable[float]]): It specifies
            the floating-point minimum and maximum values of the output tensor.
            and returns the quantized tensor.
            If it is None, it quantizes the tensor
            based on the minimum and maximum values of the original tensor
            and returns the quantized tensor.
            length of tensor_range should be 2.
            Defaults to None
        """
        self._processor.use_u8_for_output(output_name, tensor_range)

    def use_i8_for_output(self, output_name: str, tensor_range: Optional[Iterable[float]] = None):
        """
        Use model output as i8 type.
        You can directly receive the computational results of the model
        as a i8 type without converting them to f32.

        Args:
        output_name (str): Output tensor name to apply optimization
        tensor_range (Optional[Iterable[float]]): It specifies
            the floating-point minimum and maximum values of the output tensor.
            and returns the quantized tensor.
            If it is None, it quantizes the tensor
            based on the minimum and maximum values of the original tensor
            and returns the quantized tensor.
            length of tensor_range should be 2.
            Defaults to None
        """
        self._processor.use_i8_for_output(output_name, tensor_range)

    def apply_post_process(self, output_name: str, func: Callable[[np.ndarray], Any]):
        """
        Register post_process function to model output tensor

        Args:
            output_name (str): Output tensor name to apply post_process
            func (Callable[[numpy.ndarray], Any): post_process function
        """
        self._processor.apply_post_process(output_name, func)

    def apply_pre_process(self, input_name: str, func: Callable[[Any], Any]):
        """
        Register pre_process function to model input tensor.

        Args:
            input_name (str): Input tensor name to apply pre_process
            func (Callable[[Any], Any]):
                pre_process function
        """
        self._processor.apply_pre_process(input_name, func)

    def pre_process(
        self,
        input_data: Any,
    ) -> Union[np.ndarray, List[Any]]:
        """
        Run pre_process function.

        Args:
            input (Any): Input tensor to be processed
                with pre_process function

        Returns:
            output (Union[numpy.ndarray, List[Any]]):
                Output tensor result by pre_process function
        """
        return self._processor.pre_process(input_data)

    def post_process(
        self,
        input_data: List[np.ndarray],
    ) -> List[Any]:
        """
        Run post_process function.

        Args:
            input (List[numpy.ndarray]): Input tensor to be processed
                with post_process function

        Returns:
            output (List[Any]): Output tensor result by post_process function
        """
        return self._processor.post_process(input_data)


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
