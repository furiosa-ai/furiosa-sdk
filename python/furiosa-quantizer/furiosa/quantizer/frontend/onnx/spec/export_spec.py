import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

import onnx
from onnx import numpy_helper

from furiosa.quantizer.frontend.onnx.spec import spec_utils
from furiosa.quantizer.interfaces.export_spec import ExportSpec
from furiosa.quantizer.ir import spec
from furiosa.quantizer.ir.common.operator import HeightWidth, Padding

logger = logging.getLogger('Furiosa-Quantizer')
logging.basicConfig(level=logging.INFO)


class OnnxExportSpec(ExportSpec):
    def __init__(self, model: onnx.ModelProto):
        super(OnnxExportSpec).__init__()
        self.model = model
        self.tensor_shapes = self.get_tensor_shapes(self.model)
        self.initializer = {init.name: init for init in self.model.graph.initializer}
        self.attributes = self.get_attributes(self.model)

        # Build producer map
        self.producer_map: Dict[str, onnx.NodeProto] = dict()
        for node in self.model.graph.node:
            for node_output in node.output:
                if node_output in self.producer_map:
                    raise Exception(
                        "Invalid form of graph, a tensor {} has two or more producers.".format(
                            node_output
                        )
                    )
                self.producer_map[node_output] = node

        # Followings will be lazily initialized.
        self._SKIP_NODE = None
        self._MULTI_NODE_SPEC = None
        self._SINGLE_NODE_SPEC = None

    def export(self) -> Tuple[List[spec.OperatorSpec], Set[str]]:
        """
        Traverse graph and export nodes as specs.
        Returns (a list of Spec, a set of unsupported ops)
        """
        specs: List[spec.OperatorSpec] = list()
        unsupported_ops = set()

        outputs: List[str] = list(map(lambda output: output.name, self.model.graph.output))
        # To prevent traversing cyclic connections
        visited: Set[str] = set()
        visited_node: List[onnx.NodeProto] = list()

        while len(outputs) > 0:
            output = outputs.pop(0)
            if output not in self.producer_map:
                continue

            node = self.producer_map[output]
            if node.op_type in self.skip_node:
                outputs.append(node.input[0])
                visited.update([node.input[0]])
                continue

            # prevent duplicate specs from being created from nodes that have multiple outputs like Split.
            if node in visited_node:
                continue

            result = self.traverse_multi_node_spec(node)
            if result is None:
                result = self.traverse_single_node_spec(node)

            # Failed to find how to process the node
            if result is None:
                unsupported_ops.add(node.op_type)
                continue

            s, inputs = result
            # Put spec
            specs.append(s)
            # Put predecessor of node to new outputs
            outputs += list(filter(lambda input: input not in visited, inputs))
            visited.update(inputs)
            visited_node.append(node)

        return specs, unsupported_ops

    def traverse_single_node_spec(
        self, node: onnx.NodeProto
    ) -> Optional[Tuple[spec.Spec, List[str]]]:
        """
        Returns (Spec, list of inputs of the node)
        """
        if node.op_type not in self.single_node_spec:
            return None

        data_flow_input = list(
            filter(lambda input: input not in self.initializer.keys(), node.input)
        )

        return self.single_node_spec[node.op_type](node), data_flow_input

    def traverse_multi_node_spec(
        self, node: onnx.NodeProto
    ) -> Optional[Tuple[spec.Spec, List[str]]]:
        """
        Returns (Spec, list of inputs of the node)
        """
        if node.op_type not in self.multi_node_spec:
            return None

        found = None
        for func in self.multi_node_spec[node.op_type]:
            result = func(node)
            if result is None:
                continue
            # Check the ambiguity
            if found is not None:
                logger.warning(
                    "Find two or more ways of exporting as spec from multi-node for the the node {}, ".format(
                        node.op_type
                    )
                )
                return found
            found = result
        return found

    @property
    def skip_node(self) -> Set[str]:
        if self._SKIP_NODE is None:
            self._SKIP_NODE = {'Relu', 'BatchNormalization'}
        return self._SKIP_NODE

    @property
    def multi_node_spec(
        self,
    ) -> Dict[str, List[Callable[[onnx.NodeProto], Optional[Tuple[spec.Spec, List[str]]]]]]:
        if self._MULTI_NODE_SPEC is None:
            self._MULTI_NODE_SPEC = {'Div': [self.multi_node_lp_norm]}
        return self._MULTI_NODE_SPEC

    @property
    def single_node_spec(self) -> Dict[str, Callable[[onnx.NodeProto], spec.Spec]]:
        if self._SINGLE_NODE_SPEC is not None:
            return self._SINGLE_NODE_SPEC

        self._SINGLE_NODE_SPEC = {
            'Conv': self.conv2d,
            'ConvTranspose': self.convtranspose2d,
            'MaxPool': self.maxpool2d,
            'AveragePool': self.avgpool2d,
            'GlobalAveragePool': self.avgpool2d,
            'Gemm': self.gemm,
            'MatMul': self.matmul,
            'DepthToSpace': self.depthtospace,
            'Resize': self.resize,
            'Add': self.add,
            'Sub': self.sub,
            'Mul': self.mul,
            'Div': self.div,
            'Exp': self.exp,
            'Sigmoid': self.sigmoid,
            'Softplus': self.softplus,
            'Gelu': self.gelu,
            'ReduceMean': self.reduce_mean,
            'ReduceSum': self.reduce_sum,
            'ReduceL2': self.reduce_l2,
            'Squeeze': self.squeeze,
            'Unsqueeze': self.unsqueeze,
            'Reshape': self.reshape,
            'Expand': self.expand,
            'Concat': self.concatenation,
            'Transpose': self.transpose,
            'Slice': self.slice,
            'Flatten': self.flatten,
            'Pad': self.pad,
            'Split': self.split,
            'Softmax': self.softmax,
            'Clip': self.clip,
            'LayerNormalization': self.layer_norm,
            'LpNormalization': self.lp_norm,
        }
        return self._SINGLE_NODE_SPEC

    @staticmethod
    def get_tensor_shapes(model: onnx.ModelProto) -> Dict[str, Tuple[int]]:
        input_shapes = dict()
        for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            shape = [int(dim.dim_value) for dim in vi.type.tensor_type.shape.dim]
            input_shapes[vi.name] = tuple(shape)

        # include initializer's shape for this is also a node's input
        for init in model.graph.initializer:
            input_shapes[init.name] = numpy_helper.to_array(init).shape

        return input_shapes

    @staticmethod
    def get_attributes(model: onnx.ModelProto) -> Dict[str, Dict[str, int or float]]:
        attributes = dict()

        for node in model.graph.node:
            attrs = dict()

            for attr in node.attribute:
                if attr.type == 1:
                    attrs[attr.name] = attr.f
                elif attr.type == 2:
                    attrs[attr.name] = attr.i
                elif attr.type == 3:
                    attrs[attr.name] = attr.s.decode("utf-8")
                elif attr.type == 7:
                    attrs[attr.name] = attr.ints
                else:
                    raise Exception('Unknown data type: %s' % attr.type)

            attributes[node.name] = attrs

        return attributes

    def get_inputs_for_gen_spec(
        self, node: onnx.NodeProto
    ) -> Tuple[List[Tuple[int]], List[Tuple[int]], Dict]:
        input_shapes = []
        for input in node.input:
            if input == '':
                input_shapes.append([])
                continue

            input_shape = self.tensor_shapes[input]
            input_shapes.append(input_shape)

            if input in self.initializer.keys():
                continue
            assert input_shape, 'input_shape: %s. shape_inference might have failed at %s' % (
                input_shape,
                node.name,
            )

        output_shapes = []
        for output in node.output:
            output_shape = self.tensor_shapes[output]
            output_shapes.append(output_shape)
            assert output_shape, 'output_shape: %s. shape_inference might have failed at %s' % (
                output_shape,
                node.name,
            )

        attrs = self.attributes[node.name]

        return input_shapes, output_shapes, attrs

    def get_initializer_for_gen_spec(self, input_name: str) -> List[int] or List[float]:
        return numpy_helper.to_array(self.initializer[input_name]).tolist()

    def conv2d(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, output_shapes, attributes = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        output_shape = output_shapes[0]

        # TODO assert -> warning. refer to https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
        # ONNX Conv assumes n-d array as its kernel.
        assert len(attributes['kernel_shape']) == 2

        operator_spec_option = spec.Conv2d(
            input=HeightWidth(input_shape[2], input_shape[3]),
            kernel=HeightWidth(*attributes['kernel_shape']),
            stride=HeightWidth(*attributes.get('strides', (1, 1))),
            dilation=HeightWidth(*attributes.get('dilations', (1, 1))),
            batch=input_shape[0],
            input_channel=input_shape[1],
            output_channel=output_shape[1],
            groups=attributes.get('group', 1),
            padding=Padding(*attributes.get('pads', (0, 0, 0, 0))),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def convtranspose2d(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, output_shapes, attributes = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        output_shape = output_shapes[0]

        # TODO assert -> warning. refer to https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
        # ONNX Conv assumes n-d array as its kernel.
        assert len(attributes['kernel_shape']) == 2

        operator_spec_option = spec.TrasnposeConv(
            input=HeightWidth(input_shape[2], input_shape[3]),
            kernel=HeightWidth(*attributes['kernel_shape']),
            stride=HeightWidth(*attributes.get('strides', (1, 1))),
            dilation=HeightWidth(*attributes.get('dilations', (1, 1))),
            batch=input_shape[0],
            input_channel=input_shape[1],
            output_channel=output_shape[1],
            groups=attributes.get('group', 1),
            padding=Padding(*attributes.get('pads', (0, 0, 0, 0))),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def maxpool2d(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, output_shapes, attributes = self.get_inputs_for_gen_spec(node)

        assert len(input_shapes) == len(output_shapes) == 1
        input_shape = input_shapes[0]
        output_shape = output_shapes[0]

        # ONNX MaxPool assumes n-d array as its kernel.
        assert len(attributes['kernel_shape']) == 2

        operator_spec_option = spec.MaxPool2d(
            input=HeightWidth(input_shape[2], input_shape[3]),
            kernel=HeightWidth(*attributes['kernel_shape']),
            stride=HeightWidth(*attributes.get('strides', (1, 1))),
            dilation=HeightWidth(*attributes.get('dilations', (1, 1))),
            batch=input_shape[0],
            channel=output_shape[1],
            padding=Padding(*attributes.get('pads', (0, 0, 0, 0))),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def avgpool2d(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, output_shapes, attributes = self.get_inputs_for_gen_spec(node)

        assert len(input_shapes) == len(output_shapes) == 1
        input_shape = input_shapes[0]
        output_shape = output_shapes[0]

        # ONNX AveragePool assumes n-d array as its kernel.
        if node.op_type == 'AveragePool':
            assert len(attributes['kernel_shape']) == 2
        elif node.op_type == 'GlobalAveragePool':
            attributes = {'kernel_shape': (input_shape[2:])}

        operator_spec_option = spec.AveragePool2d(
            input=HeightWidth(input_shape[2], input_shape[3]),
            kernel=HeightWidth(*attributes['kernel_shape']),
            stride=HeightWidth(*attributes.get('strides', (1, 1))),
            dilation=HeightWidth(*attributes.get('dilations', (1, 1))),
            batch=input_shape[0],
            channel=output_shape[1],
            padding=Padding(*attributes.get('pads', (0, 0, 0, 0))),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def gemm(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        alpha = attributes.get('alpha', float(1.0))
        beta = attributes.get('beta', float(1.0))
        m, k, n = spec_utils.gemm_shapes(
            input_shapes, attributes.get('transA', int(0)), attributes.get('transB', int(0))
        )
        operator_spec_option = spec.Gemm(alpha=alpha, beta=beta, m=m, k=k, n=n)
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def matmul(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 2
        lhs_shape, rhs_shape = [*input_shapes[0]], [*input_shapes[1]]
        operator_spec_option = spec.MatMul(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def depthtospace(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)

        assert len(input_shapes) == 1
        input_shape = input_shapes[0]

        mode = attributes.get('mode', 'DCR')
        if mode == 'CRD':
            mode = 'ColumnRowDepth'
        elif mode == 'DCR':
            mode = 'DepthColumnRow'
        else:
            raise Exception('Unknown mode: %s. Mode must be one of "DCR" or "CRD".' % mode)

        operator_spec_option = spec.DepthToSpace(
            batch=input_shape[0],
            height=input_shape[2],
            width=input_shape[3],
            channel=input_shape[1],
            block_size=attributes['blocksize'],
            mode=mode,
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def resize(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        roi = self.get_initializer_for_gen_spec(node.input[1])
        scales = self.get_initializer_for_gen_spec(node.input[2])
        try:
            sizes = self.get_initializer_for_gen_spec(node.input[3])
        except IndexError:
            sizes = []

        operator_spec_option = spec.Resize(
            shape=[*input_shape], roi=roi, scales=scales, sizes=sizes
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def add(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        operator_spec_option = spec.Add(shape=[*input_shape])
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def sub(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        operator_spec_option = spec.Sub(shape=[*input_shape])
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def mul(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        operator_spec_option = spec.Mul(shape=[*input_shape])
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def div(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        operator_spec_option = spec.Div(shape=[*input_shape])
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def exp(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.Exp(shape=[*input_shape])
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def sigmoid(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.Sigmoid(shape=[*input_shape])
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def softplus(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.Softplus(input_shape=[*input_shape])
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def gelu(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.Gelu(shape=[*input_shape])
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def reduce_mean(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.ReduceMean(
            shape=[*input_shape],
            axes=spec_utils.implicit_axis_to_explicit([*attributes['axes']], input_shape),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def reduce_sum(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.ReduceSum(
            shape=[*input_shape],
            axes=spec_utils.implicit_axis_to_explicit([*attributes['axes']], input_shape),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def reduce_l2(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.ReduceL2(
            shape=[*input_shape],
            axes=spec_utils.implicit_axis_to_explicit([*attributes['axes']], input_shape),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def squeeze(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.Squeeze(
            shape=[*input_shape],
            axes=spec_utils.implicit_axis_to_explicit([*attributes['axes']], input_shape),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def unsqueeze(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.Unsqueeze(
            shape=[*input_shape],
            axes=spec_utils.implicit_axis_to_explicit([*attributes['axes']], input_shape),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def reshape(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, output_shapes, _ = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        output_shape = output_shapes[0]
        operator_spec_option = spec.Reshape(
            input_shape=[*input_shape], output_shape=[*output_shape]
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def expand(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, output_shapes, _ = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        output_shape = output_shapes[0]
        operator_spec_option = spec.Expand(input_shape=[*input_shape], output_shape=[*output_shape])
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def concatenation(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        operator_spec_option = spec.Concatenation(
            tensors=list(map(list, input_shapes)),
            axis=spec_utils.implicit_axis_to_explicit(attributes['axis'], input_shapes[0]),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def transpose(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.Transpose(
            shape=[*input_shape],
            permutation=spec_utils.implicit_axis_to_explicit([*attributes['perm']], input_shape),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def slice(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        starts = self.get_initializer_for_gen_spec(node.input[1])
        axes = self.get_initializer_for_gen_spec(node.input[3])
        operator_spec_option = spec.Slice(
            shape=[*input_shape], offset=spec_utils.slice_offset_dict(starts, axes, input_shape)
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def flatten(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.Flatten(
            shape=[*input_shape],
            axis=spec_utils.implicit_axis_to_explicit(attributes['axis'], input_shape),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def pad(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]
        assert len(input_shape) == 4
        pads = self.get_initializer_for_gen_spec(node.input[1])
        operator_spec_option = spec.Pad(shape=[*input_shape], pad=spec_utils.horizontal_pads(*pads))
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def layer_norm(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        operator_spec_option = spec.LayerNorm(
            input_shape=[*input_shapes[0]], eps=attributes['epsilon']
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def split(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        operator_spec_option = spec.Split(
            shape=[*input_shape],
            split=[*attributes['split']],
            axis=spec_utils.implicit_axis_to_explicit(attributes.get('axis', 0), input_shape),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def softmax(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, attributes = self.get_inputs_for_gen_spec(node)
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]

        operator_spec_option = spec.Softmax(
            input_shape=[*input_shape],
            beta=attributes.get('beta', float(1.0)),
            axis=spec_utils.implicit_axis_to_explicit(attributes['axis'], input_shape),
        )
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def clip(self, node: onnx.NodeProto) -> spec.Spec:
        input_shapes, _, _ = self.get_inputs_for_gen_spec(node)
        input_shape = input_shapes[0]

        kwargs = {}
        if node.attribute:
            for attr in node.attribute:
                if attr.name == "min":
                    kwargs['min'] = float(attr.f)
                elif attr.name == "max":
                    kwargs['max'] = float(attr.f)
        else:
            assert len(node.input) == 3
            for idx, node_input in enumerate(node.input):
                if idx == 1:
                    try:
                        kwargs['min'] = float(numpy_helper.to_array(self.initializer[node_input]))
                    except KeyError:
                        kwargs['min'] = None

                elif idx == 2:
                    try:
                        kwargs['max'] = float(numpy_helper.to_array(self.initializer[node_input]))
                    except KeyError:
                        kwargs['max'] = None

        if not kwargs:
            raise Exception('Empty min and/or max.')

        operator_spec_option = spec.Clip(input_shape=[*input_shape], **kwargs)
        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def lp_norm(self, node: onnx.NodeProto) -> spec.Spec:
        input_shape, _, attrs = self.get_inputs_for_gen_spec(node)
        operator_spec_option = spec.LpNorm(input_shape=[*input_shape], **attrs)

        return spec.Spec(spec_utils.node_identifier(node), operator_spec_option)

    def multi_node_lp_norm(self, node: onnx.NodeProto) -> Optional[Tuple[spec.Spec, List[str]]]:
        """
        Starts from 'Div', traverse up to find the form of l2norm.
        Returns all inputs of l2norm, consist of multi node

        LpNormalization is not defined in ONNX Operator spec, so that we should traverse the graph:

        Input --> ReduceL2 --> Clip --> Expand --> D
              -----------------------------------> iv --> Output
        """
        inputs_of_lp_norm: List[str] = []
        for input in node.input:
            # exclude input from initializer
            if input not in self.producer_map:
                continue

            prev_node = self.producer_map[input]
            if prev_node.op_type != 'Expand':
                continue

            pprev_node = self.producer_map[prev_node.input[0]]
            if pprev_node.op_type != 'Clip':
                continue

            ppprev_node = self.producer_map[pprev_node.input[0]]
            if ppprev_node.op_type != 'ReduceL2':
                continue
            p = 2

            inputs_of_lp_norm.append(ppprev_node.input[0])
            input_shapes, _, attributes = self.get_inputs_for_gen_spec(ppprev_node)
            axis = attributes['axes'][0]

            operator_spec_option = spec.LpNorm(input_shape=[*input_shapes[0]], p=p, axis=axis)
            return (
                spec.Spec(spec_utils.node_identifier(node), operator_spec_option),
                inputs_of_lp_norm,
            )
