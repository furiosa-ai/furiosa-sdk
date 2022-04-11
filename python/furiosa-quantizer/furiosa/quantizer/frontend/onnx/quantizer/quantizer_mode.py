from typing import Optional

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model


class DFGImportable:
    def __init__(self, model, raw_data):
        copy_model = onnx.ModelProto()
        copy_model.CopyFrom(model)
        self.model = copy_model
        self.raw_data = raw_data

        self.node = {node.name: node for node in self.model.graph.node}
        self.node_by_output = {
            node_output: node for node in self.model.graph.node for node_output in node.output
        }
        self.node_by_input = {
            node_input: node for node in self.model.graph.node for node_input in node.input
        }
        self.initializer = {init.name: init for init in self.model.graph.initializer}
        self.graph_input = {vi.name: vi for vi in self.model.graph.input}
        self.value_info = {vi.name: vi for vi in self.model.graph.value_info}

    def transform(self):
        self.remove_quantizelinear_operator_with_initializer()
        self.transform_to_integer_arithmetic_operator()

        return self.model

    def remove_quantizelinear_operator_with_initializer(self):
        new_nodes = []
        rm_nodes = []
        for node in self.model.graph.node:
            if node.op_type != 'QuantizeLinear':
                new_nodes.append(node)
                continue
            if node.input[0] not in self.initializer:
                new_nodes.append(node)
                continue

            # node.input[0] to be removed from model.graph.initializer
            init = self.initializer.pop(node.input[0])

            # quantize initializer
            s = self.initializer[node.input[1]]
            zp = self.initializer[node.input[2]]

            dequant_linear_output = node.output[0].split("_quantized")[0] + "_dequantized"
            middle_node = self.node_by_input[dequant_linear_output]

            # gives output_channel_axis for per-channel quantized Conv/ConvTranspose weight/bias
            rank = len(init.dims)
            if rank > 1:
                if middle_node.op_type == "Conv":
                    output_channel_axis = 0
                elif middle_node.op_type == 'ConvTranspose':
                    output_channel_axis = 1
                else:
                    output_channel_axis = None
            else:
                output_channel_axis = None

            quantized_data = self._quantize_data(init, s, zp, output_channel_axis)

            # node.output[0] to be updated to model.graph.initializer instead
            flattened = quantized_data.flatten()
            if self.raw_data:
                flattened = flattened.tobytes()

            self.initializer.update(
                {
                    node.output[0]: onnx.helper.make_tensor(
                        node.output[0], zp.data_type, init.dims, flattened, raw=self.raw_data
                    )
                }
            )

            # node.output[0] to be removed from model.graph.value_info
            self.value_info.pop(node.output[0])

            rm_nodes.append(node)

        self.model = utils.rebuild_model(
            model=self.model,
            new_nodes=[node for node in new_nodes if node not in rm_nodes],
            eliminate=False,
            renaming=False,
        )

        self._update_graph_field(field='initializer', proto=self.initializer.values())
        self._update_graph_field(field='value_info', proto=self.value_info.values())
        self._update_graph_field(field='input', proto=self.graph_input.values())

        check_model(self.model, check_runnable=False)

    def transform_to_integer_arithmetic_operator(self):
        new_nodes = []
        rm_nodes = []

        for node in self.model.graph.node:
            if node.op_type not in ['Conv', 'MatMul']:
                new_nodes.append(node)
                continue

            node_i0 = self.node_by_output[node.input[0]]
            node_i1 = self.node_by_output[node.input[1]]
            node_o0 = self.node_by_input[node.output[0]]
            rm_nodes.extend([node_i0, node_i1, node_o0])

            # When a tensor is used as model's output, it is not in the value_info.
            # 'None' argument to handle the case.
            self.value_info.pop(node_i0.output[0], None)
            self.value_info.pop(node_i1.output[0], None)
            self.value_info.pop(node_o0.input[0], None)

            node_i2 = None
            if len(node.input) == 3:
                node_i2 = self.node_by_output[node.input[2]]
                rm_nodes.append(node_i2)

                self.value_info.pop(node_i2.output[0], None)

            rm_nodes.extend([node])
            new_nodes.append(
                self._make_integer_arithmetic_operator(node, node_i0, node_i1, node_o0, node_i2)
            )

        self.model = utils.rebuild_model(
            model=self.model,
            new_nodes=[node for node in new_nodes if node not in rm_nodes],
            eliminate=False,
        )

        self._update_graph_field(field='value_info', proto=self.value_info.values())

        check_model(self.model, check_runnable=False)

    @staticmethod
    def _make_integer_arithmetic_operator(node, node_i0, node_i1, node_o0, node_i2=None):
        quant_op_type = f'QLinear{node.op_type}'

        quant_inputs = [*node_i0.input, *node_i1.input, *node_o0.input[1:]]

        if node_i2:
            quant_inputs.append(node_i2.input[0])

        quant_outputs = [node_o0.output[0]]

        attr_kwargs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}

        quant_node = onnx.helper.make_node(
            quant_op_type, quant_inputs, quant_outputs, **attr_kwargs
        )

        return quant_node

    @staticmethod
    def _quantize_data(
        data: onnx.TensorProto,
        scale: onnx.TensorProto,
        zero_point: onnx.TensorProto,
        axis: Optional[int] = None,
    ) -> np.ndarray:
        data_arr = np.atleast_1d(onnx.numpy_helper.to_array(data)).astype(np.float32)
        scale_arr = np.atleast_1d(onnx.numpy_helper.to_array(scale)).astype(np.float32)
        zero_point_arr = np.atleast_1d(onnx.numpy_helper.to_array(zero_point)).astype(np.float32)

        if axis is not None:
            new_axes = list(range(data_arr.ndim))
            new_axes.pop(axis)
            scale_arr = np.expand_dims(scale_arr, axis=new_axes)
            zero_point_arr = np.expand_dims(zero_point_arr, axis=new_axes)

        quantized_data = np.round(data_arr / scale_arr) + zero_point_arr

        np_dtype_info = np.iinfo(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[zero_point.data_type])
        np_dtype_info_min = np_dtype_info.min
        np_dtype_info_max = np_dtype_info.max

        return np.clip(quantized_data, np_dtype_info_min, np_dtype_info_max).astype(
            onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[zero_point.data_type]
        )

    def _update_graph_field(self, field, proto):
        self.model.graph.ClearField(field)
        getattr(self.model.graph, field).extend(proto)


class ONNXRuntimeExecutable(DFGImportable):
    def transform(self):

        self._remove_quant_dequantlinear_operator_with_initializer()

        return self.model

    def _remove_quant_dequantlinear_operator_with_initializer(self):
        rm_nodes = []
        new_nodes = []
        for node in self.model.graph.node:
            if node.op_type != 'QuantizeLinear':
                new_nodes.append(node)
                continue

            if node.input[0] not in self.initializer:
                new_nodes.append(node)
                continue

            rm_nodes.append(node)
            rm_nodes.extend(
                [
                    dequant_node
                    for dequant_node in self.model.graph.node
                    if dequant_node.op_type == 'DequantizeLinear'
                    if dequant_node.input[0] == node.output[0]
                ]
            )

        for node in self.model.graph.node:
            if node.op_type in ('QuantizeLinear', 'DequantizeLinear'):
                continue

            for idx, node_input in enumerate(node.input):
                if '_dequantized' not in node_input:
                    continue

                init_name = node_input.split('_dequantized')[0]
                if init_name not in self.initializer:
                    continue

                node.input[idx] = init_name + '_fake_quantized'

                init = self.initializer.pop(init_name)
                s = self.initializer.pop(init_name + '_scale')
                zp = self.initializer.pop(init_name + '_zero_point')

                # gives output_channel_axis for per-channel quantized Conv/ConvTranspose weight/bias
                rank = len(init.dims)
                if rank > 1:
                    if node.op_type == "Conv":
                        output_channel_axis = 0
                    elif node.op_type == 'ConvTranspose':
                        output_channel_axis = 1
                    else:
                        output_channel_axis = None
                else:
                    output_channel_axis = None

                fake_quantized_data = self._fake_quantize_data(init, s, zp, output_channel_axis)

                flattened = fake_quantized_data.flatten()
                if self.raw_data:
                    flattened = flattened.tobytes()
                self.initializer.update(
                    {
                        init_name
                        + '_fake_quantized': onnx.helper.make_tensor(
                            name=init_name + '_fake_quantized',
                            data_type=onnx.TensorProto.FLOAT,
                            dims=init.dims,
                            vals=flattened,
                            raw=self.raw_data,
                        )
                    }
                )

        self.model = utils.rebuild_model(
            model=self.model,
            new_nodes=[node for node in new_nodes if node not in rm_nodes],
            eliminate=True,
        )
        self._update_graph_field(field='initializer', proto=self.initializer.values())

        check_model(self.model, check_runnable=True)

    @staticmethod
    def _dequantize_data(
        data: onnx.TensorProto,
        scale: onnx.TensorProto,
        zero_point: onnx.TensorProto,
        axis: Optional[int] = None,
    ) -> np.ndarray:
        data_arr = np.atleast_1d(onnx.numpy_helper.to_array(data)).astype(np.float32)
        scale_arr = np.atleast_1d(onnx.numpy_helper.to_array(scale)).astype(np.float32)
        zero_point_arr = np.atleast_1d(onnx.numpy_helper.to_array(zero_point)).astype(np.float32)

        if axis is not None:
            new_axes = list(range(data_arr.ndim))
            new_axes.pop(axis)
            scale_arr = np.expand_dims(scale_arr, axis=new_axes)
            zero_point_arr = np.expand_dims(zero_point_arr, axis=new_axes)

        return (data_arr - zero_point_arr) * scale_arr

    def _fake_quantize_data(
        self,
        data: onnx.TensorProto,
        scale: onnx.TensorProto,
        zero_point: onnx.TensorProto,
        axis: Optional[int] = None,
    ) -> np.ndarray:
        quantized_data = self._quantize_data(data, scale, zero_point, axis)
        flattened = quantized_data.flatten()
        if self.raw_data:
            flattened = flattened.tobytes()
        dequantized_data = self._dequantize_data(
            onnx.helper.make_tensor(
                name=data.name,
                data_type=zero_point.data_type,
                dims=data.dims,
                vals=flattened,
                raw=self.raw_data,
            ),
            scale,
            zero_point,
            axis,
        )

        return dequantized_data
