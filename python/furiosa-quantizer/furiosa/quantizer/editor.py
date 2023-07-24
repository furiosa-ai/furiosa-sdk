import enum
from typing import Optional, Tuple

import furiosa_quantizer_impl
import onnx


class TensorType(enum.IntEnum):
    """TensorType.

    TensorType which specifies type of the data.
    It is used for ModelEditor.retype_as
    """

    U8 = 1
    I8 = 2


class ModelEditor:
    """ModelEditor.

    Apply additional optimization information for
    ONNX Model.
    This will record optimization settings in ONNX
    model metadata
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
        self._input = model.graph.input
        self._output = model.graph.output
        self._processor = furiosa_quantizer_impl.Processor(model)

    def retype_as(
        self,
        tensor_name: str,
        tensor_type: TensorType,
        tensor_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Retype model input(or output) as i8(or u8) type.
        if there is a existed same key in ONNX metadata_props already,
        Remove existed ONNX metadata

        Args:
        tensor_name (str): Tensor name to retype
        tensor_type: TensorType: DataType to be applied in Tensor
        tensor_range (Optional[Iterable[float]]): It specifies
            the floating-point minimum and maximum values of the output tensor.
            and returns the quantized tensor.
            If it is None, it quantizes the tensor
            based on the minimum and maximum values of the original tensor
            and returns the quantized tensor.
            length of tensor_range should be 2.
            Defaults to None
        """
        # for input
        for vi in self._input:
            if vi.name == tensor_name:
                if tensor_range is not None:
                    raise ValueError(
                        f"We did not support retype input with tensor range({tensor_range})"
                    )
                if tensor_type == TensorType.U8:
                    self._processor.use_u8_for_input(tensor_name)
                    return
                if tensor_type == TensorType.I8:
                    raise ValueError("We did not surpport retype input as i8 type")

        # for output
        for vi in self._output:
            if vi.name == tensor_name:
                if tensor_type == TensorType.U8:
                    self._processor.use_u8_for_output(tensor_name, tensor_range)
                    return
                if tensor_type == TensorType.I8:
                    self._processor.use_i8_for_output(tensor_name, tensor_range)
                    return
        raise ValueError(f"can't find any in/output tensor which name is {tensor_name}")
