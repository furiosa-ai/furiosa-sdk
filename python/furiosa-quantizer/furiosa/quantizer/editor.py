import enum
from typing import List, Optional, Tuple

import furiosa_quantizer_impl
import onnx


class TensorType(enum.IntEnum):
    """An enumeration class representing the element type of a tensor.

    This class is used with the ModelEditor.convert_{input,output}_type method to specify the
    desired element type.
    """

    UINT8 = 1
    INT8 = 2


def get_pure_input_names(model: onnx.ModelProto) -> List[str]:
    """Return the names of inputs in an ONNX model that have no associated initializers.

    Args:
        model (onnx.ModelProto): An ONNX model.

    Returns:
        List[str]: A list of the names of inputs in the model that have no associated initializers.
    """
    initializers = {tensor.name for tensor in model.graph.initializer}
    return [
        value_info.name for value_info in model.graph.input if value_info.name not in initializers
    ]


def get_output_names(model: onnx.ModelProto) -> List[str]:
    """Return the names of outputs in an ONNX model.

    Args:
        model (onnx.ModelProto): An ONNX model.

    Returns:
        List[str]: A list of the names of outputs in the model.
    """
    return [value_info.name for value_info in model.graph.output]


class ModelEditor:
    """A utility class for manipulating ONNX models."""

    def __init__(self, model: onnx.ModelProto):
        self._input = set(get_pure_input_names(model))
        self._output = set(get_output_names(model))
        self._processor = furiosa_quantizer_impl.Processor(model)

    def convert_input_type(self, tensor_name: str, tensor_type: TensorType) -> None:
        """Convert the element type of an input tensor named tensor_name to tensor_type.

        Args:
            tensor_name (str): The name of an input tensor to convert.
            tensor_type (TensorType): The desired element type.
        """
        if tensor_name not in self._input:
            raise ValueError(f"Could find a pure input tensor named '{tensor_name}'.")

        if tensor_type == TensorType.UINT8:
            self._processor.use_u8_for_input(tensor_name)
        else:
            raise ValueError(f"ModelEditor.convert_input_type does not support {tensor_type} yet.")

    def convert_output_type(
        self,
        tensor_name: str,
        tensor_type: TensorType,
        tensor_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Convert the element type of an output tensor named tensor_name to tensor_type.

        Args:
            tensor_name (str): The name of an output tensor to convert.
            tensor_type (TensorType): The desired element type.
            tensor_range (Optional[Tuple[float, float]]): A new min/max range of the output tensor.
                If it is None, the original range will be retained. Defaults to None.
        """
        if tensor_name not in self._output:
            raise ValueError(f"Could not find an output tensor named '{tensor_name}'.")

        if tensor_type == TensorType.UINT8:
            self._processor.use_u8_for_output(tensor_name, tensor_range)
        elif tensor_type == TensorType.INT8:
            self._processor.use_i8_for_output(tensor_name, tensor_range)
        else:
            raise ValueError(f"ModelEditor.convert_output_type does not support {tensor_type} yet.")
