from typing import Any, Callable, Iterable, List, Optional, Union
import onnx
import numpy as np

import furiosa_quantizer_impl


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
