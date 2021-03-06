from typing import List, Tuple, Dict

import warnings

import onnx

import numpy as np

from onnx import TensorProto, TensorAnnotation, StringStringEntryProto
import onnxruntime as ort
from onnxruntime_tools.quantization.quantize import _attribute_to_kwarg

__PRODUCER__ = "jason_furiosa"

# This OP list is based on ONNX operator spec.
# Integer_math and Integer_non_math ops can handle int8 inputs whereas Sandwich ops can not.
# Note: Concat is exceptional. As it requires re-quantizing multiple inputs.
__DYNAMIC_RANGE_COLLECTORS__ = {
    'integer_math': [
        'Conv',
        'MatMul',
    ],
    'integer_non_math': [
        'MaxPool',
        'Squeeze',
        'Unsqueeze',
        'Gather',
        'Transpose',
        'Reshape',
        'DepthToSpace',
        'Expand',
        'Clip',
        'Split',
        'Pad',
        'Resize',
        'Flatten',
        'Slice',
    ],
    'sandwich': [
        'Gemm',
        'Add',
        'ReduceMean',
        'Softmax',
        'Relu',
        'Concat',
        'Softmax',
        'Softplus',
        'ReduceL2',
        'LayerNormalization',
        'Gelu',
        'GlobalAveragePool',
        'Sigmoid',
        'Mul',
        'AveragePool',
        'ReduceSum',
        'Div',
        'ConvTranspose',
        'LpNormalization',
    ],
}


class QuantizationMode:
    # dfg: Quantize graph to DFG(Quantized graph) export
    dfg = 0
    # fake: Evaluate quantized graph replacing QConvLinear with Conv2d/MatMul &
    fake = 1


def get_qrange(qtype):
    """
    source: onnxruntime quantization tools
    """
    if qtype == TensorProto.UINT8:
        return 255  # 2^b - 1
    elif qtype == TensorProto.INT8:
        return 254  # [-(2^{b-1}-1), 2^{b-1}-1]: [-127, 127] for 8 bits.
    else:
        raise ValueError('unsupported quantization data type')


def get_vi_dtype(vi):
    """
    This function returns value_info's data type

    :param vi: graph.value_info
    :return: graph.value_info.type.tensor_type.elem_type
    """
    return vi.type.tensor_type.elem_type


def is_float_tensor(vi):
    if get_vi_dtype(vi) == onnx.TensorProto.FLOAT:
        return True
    return False


def activation_scale_zeropoint(rmin, rmax, input_qtype):
    rmin = min(rmin, 0)
    rmax = max(rmax, 0)

    return asymmetric_scale_zeropoint(rmin, rmax, input_qtype)


def asymmetric_scale_zeropoint(rmin, rmax, input_qtype):
    """
    source: onnxruntime quantization tools
    """
    scale = np.float32((rmax - rmin) / 255 if rmin != rmax else 1)
    # The minimum positive (subnormal) value is 2 ** -149 for IEEE 754 single-precision binary floating-point format
    # source: https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Exponent_encoding
    scale = max(scale, 2 ** -149)
    if input_qtype == TensorProto.UINT8:
        initial_zero_point = (0 - rmin) / scale
        zero_point = np.uint8(round(max(0, min(255, initial_zero_point))))
        return np.array(zero_point), np.array(scale)
    elif input_qtype == TensorProto.INT8:
        initial_zero_point = -128 - rmin / scale
        zero_point = np.int8(round(max(-128, min(127, initial_zero_point))))
        return np.array(zero_point), np.array(scale)
    else:
        Exception('qType must be one of UINT8 or INT8')


def calculate_activation_quant_params(dynamic_ranges: Dict,
                                      node_list: List[onnx.NodeProto],
                                      input_qtype: TensorProto,
                                      value_info: Dict) -> Dict:
    quantization_params = {}
    for node in node_list:
        # Quantize activation input/output, following TFLite's quantization specification
        if node.op_type in ['MaxPool', 'Squeeze', 'Unsqueeze', 'Gather', 'Transpose', 'Reshape',
                            'DepthToSpace', 'Expand', 'Flatten', 'GlobalAveragePool', 'AveragePool']:
            if not is_float_tensor(value_info[node.input[0]]):
                continue
            if node.input[0] not in quantization_params.keys():
                quantization_params[node.input[0]] = activation_scale_zeropoint(
                    *dynamic_ranges[node.input[0]],
                    input_qtype)

            quantization_params[node.output[0]] = quantization_params[node.input[0]]
        elif node.op_type in ['Softmax', 'Sigmoid']:
            if node.input[0] not in quantization_params.keys():
                quantization_params[node.input[0]] = activation_scale_zeropoint(
                    *dynamic_ranges[node.input[0]],
                    input_qtype)

            if input_qtype == TensorProto.INT8:
                zero_point = np.array((np.int8(-128)))
            elif input_qtype == TensorProto.UINT8:
                zero_point = np.array(np.uint8(0))
            else:
                raise Exception()
            quantization_params[node.output[0]] = (zero_point, np.array(np.float32(1.0 / 256.0)))
        elif node.op_type in ['LpNormalization']:
            if node.input[0] not in quantization_params.keys():
                quantization_params[node.input[0]] = activation_scale_zeropoint(
                    *dynamic_ranges[node.input[0]],
                    input_qtype)

            if input_qtype == TensorProto.INT8:
                zero_point = np.array((np.int8(0)))
            elif input_qtype == TensorProto.UINT8:
                zero_point = np.array(np.uint8(128))
            else:
                raise Exception()
            quantization_params[node.output[0]] = (zero_point, np.array(np.float32(1.0 / 128.0)))
        else:
            for name in list(node.input) + list(node.output):
                if name not in dynamic_ranges:
                    continue
                if name in quantization_params.keys():
                    continue
                rmin, rmax = dynamic_ranges[name]
                zero_point, scale = activation_scale_zeropoint(rmin, rmax, input_qtype)
                quantization_params[name] = (zero_point, scale)

    return quantization_params


def calculate_weight_quant_params(data: np.array, weight_qtype: TensorProto) -> Tuple[int, float]:
    """
        :parameter data: data to quantize
        :parameter weight_qtype: quantization data type of weight
        :return: quantized weights, zero point, scale

        To pack weights, we compute a linear transformation
            - when data type == uint8 mode, from [rmin, rmax] -> [0, 2^{b-1}] and
            - when data type == int8, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
                m = max(abs(rmin), abs(rmax))

        and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation
        r = S(q-z), where
            r: real original value
            q: quantized value
            S: scale
            z: zero point

        source: onnxruntime quantization tools
    """
    rmin = min(min(data), 0)
    rmax = max(max(data), 0)

    quantized_range = get_qrange(weight_qtype)

    if weight_qtype == TensorProto.INT8:
        max_range = max(abs(rmin), abs(rmax))
        if max_range > 0:
            scale = (max_range * 2.0) / quantized_range
        else:
            warnings.warn('both the min and the max of data are 0')
            scale = 1.0
        zero_point = 0
    elif weight_qtype == TensorProto.UINT8:
        scale = (float(rmax) - rmin) / quantized_range if rmin != rmax else 1
        zero_point = round((0 - rmin) / scale)  # round to nearest integer
    else:
        raise ValueError(
            "Unexpected data type {} requested. Only INT8 and UINT8 are supported.".format(weight_qtype))

    # The minimum positive (subnormal) value is 2 ** -149 for IEEE 754 single-precision binary floating-point format
    # source: https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Exponent_encoding
    scale = max(scale, 2 ** -149)
    return zero_point, scale


def make_tensor_annotation(tensor_name, zero_point_name, scale_name):
    '''
    Helper function to make_tensor_annotation
    '''
    annot = TensorAnnotation()
    annot.tensor_name = tensor_name

    quant_param_scale = StringStringEntryProto()
    quant_param_scale.key = 'SCALE_TENSOR'
    quant_param_scale.value = scale_name

    quant_param_zero_point = StringStringEntryProto()
    quant_param_zero_point.key = 'ZERO_POINT_TENSOR'
    quant_param_zero_point.value = zero_point_name

    annot.quant_parameter_tensor_names.extend([quant_param_scale, quant_param_zero_point])

    return annot


def attribute_to_kwargs(attributes: onnx.AttributeProto) -> Dict:
    kwargs = {}
    for attr in attributes:
        kwargs.update(_attribute_to_kwarg(attr))

    return kwargs


def append_suffix(name: str, suffix: List[str]) -> List[str]:
    """
    Helper function to append suffixes to the given name.
    """
    return list(map(lambda x: name + x, suffix))


def get_input_tensors(model: onnx.ModelProto) -> List[str]:
    ort.set_default_logger_severity(3)
    sess = ort.InferenceSession(model.SerializeToString())
    input_tensors = [inp.name for inp in sess.get_inputs()]

    return input_tensors
