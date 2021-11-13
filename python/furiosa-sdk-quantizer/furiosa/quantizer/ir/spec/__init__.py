from typing import Dict, List, Optional

from furiosa.quantizer.ir.common.operator import HeightWidth, HorizontalPadding, Padding


class Spec:
    def __init__(self, name: str, operator_spec: "OperatorSpec"):
        self.name = name
        self.option = operator_spec

    def kind(self):
        raise NotImplementedError()

    def as_dict(self) -> Dict[str, any]:
        return {
            'name': self.name,
            'option': {
                # quantizer only supports 'OperatorSpec'.
                'Operator': self.option.as_dict(),
            },
        }


class OperatorSpec:
    def kind(self):
        raise NotImplementedError()

    def as_dict(self) -> Dict[str, any]:
        return {
            self.kind(): dict(map(lambda item: self._handle_nested_spec(*item), vars(self).items()))
        }

    @staticmethod
    def _handle_nested_spec(k, v):
        if hasattr(v, 'as_dict'):
            return k, getattr(v, 'as_dict')()
        else:
            return k, v


class PaddingSpecCustom:
    def __init__(self, padding: Padding):
        self.Custom = padding


class Conv2d(OperatorSpec):
    def kind(self):
        return 'Conv2d'

    def __init__(
        self,
        input: HeightWidth,
        kernel: HeightWidth,
        stride: HeightWidth,
        dilation: HeightWidth,
        batch: int,
        input_channel: int,
        output_channel: int,
        groups: int,
        padding: Padding,
    ):
        self.input = input
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.batch = batch
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.groups = groups
        self.padding_spec = PaddingSpecCustom(padding)


class TrasnposeConv(Conv2d):
    def kind(self):
        return 'TransposeConv'


class MaxPool2d(OperatorSpec):
    def kind(self):
        return 'MaxPool2d'

    def __init__(
        self,
        input: HeightWidth,
        kernel: HeightWidth,
        stride: HeightWidth,
        dilation: HeightWidth,
        batch: int,
        channel: int,
        padding: Padding,
    ):
        self.input = input
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.batch = batch
        self.channel = channel
        self.padding_spec = PaddingSpecCustom(padding)


class AveragePool2d(OperatorSpec):
    def kind(self):
        return 'AveragePool2d'

    def __init__(
        self,
        input: HeightWidth,
        kernel: HeightWidth,
        stride: HeightWidth,
        dilation: HeightWidth,
        batch: int,
        channel: int,
        padding: Padding,
    ):
        self.input = input
        self.kernel = kernel
        self.stride = stride
        self.batch = batch
        self.channel = channel
        self.dilation = dilation
        self.padding_spec = PaddingSpecCustom(padding)


class Gemm(OperatorSpec):
    def kind(self):
        return 'Gemm'

    def __init__(self, alpha: float, beta: float, m: int, k: int, n: int):
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.k = k
        self.n = n


class MatMul(OperatorSpec):
    def kind(self):
        return 'MatMul'

    def __init__(self, lhs_shape: List[int], rhs_shape: List[int]):
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape


class DepthToSpace(OperatorSpec):
    def kind(self):
        return 'DepthToSpace'

    def __init__(
        self, batch: int, height: int, width: int, channel: int, block_size: int, mode: str
    ):
        self.batch = batch
        self.height = height
        self.width = width
        self.channel = channel
        self.block_size = block_size
        self.mode = mode


class Resize(OperatorSpec):
    def kind(self):
        return 'Resize'

    def __init__(self, shape: List[int], roi: List[int], scales: List[float], sizes: List[int]):
        self.shape = shape
        self.roi = roi
        self.scales = scales
        self.sizes = sizes


class Add(OperatorSpec):
    def kind(self):
        return 'Add'

    def __init__(self, shape: List[int]):
        self.shape = shape


class Sub(Add):
    def kind(self):
        return 'Sub'


class Mul(Add):
    def kind(self):
        return 'Mul'


class Div(Add):
    def kind(self):
        return 'Div'


class Exp(Add):
    def kind(self):
        return 'Exp'


class Sigmoid(Add):
    def kind(self):
        return 'Sigmoid'


class Softplus(OperatorSpec):
    def kind(self):
        return 'Softplus'

    def __init__(self, input_shape: List[int]):
        self.input_shape = input_shape


class Gelu(Add):
    def kind(self):
        return 'Gelu'


class ReduceMean(OperatorSpec):
    def kind(self):
        return 'ReduceMean'

    def __init__(self, shape: List[int], axes: List[int]):
        self.shape = shape
        self.axes = axes


class ReduceSum(ReduceMean):
    def kind(self):
        return 'ReduceSum'


class ReduceL2(ReduceMean):
    def kind(self):
        return 'ReduceL2'


class Squeeze(ReduceMean):
    def kind(self):
        return 'Squeeze'


class Unsqueeze(ReduceMean):
    def kind(self):
        return 'Unsqueeze'


class Reshape(OperatorSpec):
    def kind(self):
        return 'Reshape'

    def __init__(self, input_shape: List[int], output_shape: List[int]):
        self.input_shape = input_shape
        self.output_shape = output_shape


class Expand(Reshape):
    def kind(self):
        return 'Expand'


class Concatenation(OperatorSpec):
    def kind(self):
        return 'Concatenation'

    def __init__(self, tensors: List[List[int]], axis: int):
        self.tensors = tensors
        self.axis = axis


class Transpose(OperatorSpec):
    def kind(self):
        return 'Transpose'

    def __init__(self, shape: List[int], permutation: List[int]):
        self.shape = shape
        self.permutation = permutation


class Slice(OperatorSpec):
    def kind(self):
        return 'Slice'

    def __init__(self, shape: List[int], offset: List[int]):
        self.shape = shape
        self.offset = offset


class Flatten(OperatorSpec):
    def kind(self):
        return 'Flatten'

    def __init__(self, shape: List[int], axis: int):
        self.shape = shape
        # The field `Axis` isn't in npu-tools.
        self.axis = axis


class Pad(OperatorSpec):
    def kind(self):
        return 'Pad'

    def __init__(self, shape: List[int], pad: List[HorizontalPadding]):
        self.shape = shape
        self.pad = pad


class Split(OperatorSpec):
    def kind(self):
        return 'Split'

    def __init__(self, shape: List[int], split: List[int], axis: int):
        self.shape = shape
        self.split = split
        self.axis = axis


class Softmax(OperatorSpec):
    def kind(self):
        return 'Softmax'

    def __init__(self, input_shape: List[int], beta: float, axis: int):
        self.input_shape = input_shape
        self.beta = beta
        self.axis = axis


class Clip(OperatorSpec):
    def kind(self):
        return 'Clip'

    def __init__(
        self, input_shape: List[int], min: Optional[float] = None, max: Optional[float] = None
    ):
        self.input_shape = input_shape
        self.min = min
        self.max = max


class LpNorm(OperatorSpec):
    def kind(self):
        return 'LpNorm'

    def __init__(self, input_shape: List[int], p: int, axis: int):
        self.input_shape = input_shape
        self.p = p
        self.axis = axis


class LayerNorm(OperatorSpec):
    def kind(self):
        return 'LayerNorm'

    def __init__(self, input_shape: List[int], eps: float):
        self.input_shape = input_shape
        self.eps = eps
