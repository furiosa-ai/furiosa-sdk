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

