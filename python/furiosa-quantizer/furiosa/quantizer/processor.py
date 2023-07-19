import enum
from typing import Optional, Tuple
import onnx

import furiosa_quantizer_impl


class TensorType(enum.IntEnum):
    U8 = 1
    I8 = 2


def retype_as(
    model: onnx.ModelProto,
    tensor_name: str,
    tensor_type: TensorType,
    tensor_range: Optional[Tuple[float, float]] = None,
):
    """
    Retype model input(or output) as i8(or u8) type.
    These information will be recorded in ONNX metadata_props.
    if there is a existed same key in ONNX metadata_props already,
    Remove existed ONNX metadata

    Args:
    model (onnx.ModelProto): An ONNX ModelProto
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
    processor = furiosa_quantizer_impl.Processor(model)
    # for input
    for vi in model.graph.inputs:
        if vi.name == tensor_name:
            if tensor_range is not None:
                raise ValueError(
                    f"We did not surpport retype input with tensor range({tensor_range})"
                )
            if tensor_type == TensorType.U8:
                processor.use_u8_for_input(tensor_name)
                return
            if tensor_type == TensorType.I8:
                raise ValueError("We did not surpport retype input as i8 type")

    # for output
    for vi in model.graph.outputs:
        if vi.name == tensor_name:
            if tensor_type == TensorType.U8:
                processor.use_u8_for_output(tensor_name, tensor_range)
                return
            if tensor_type == TensorType.I8:
                processor.use_i8_for_output(tensor_name, tensor_range)
                return
