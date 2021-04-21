# -------------------------------------------------------------------------
#
# FuriosaAI CONFIDENTIAL
# __________________
#
# FuriosaAI Incorporated, All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of FuriosaAI and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to FuriosaAI Inc
# and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from FuriosaAI Inc.
# --------------------------------------------------------------------------
from typing import List, Dict, Tuple, Optional

import os
import onnx
import tqdm
import numpy as np

from onnx import numpy_helper
from onnx.helper import make_node, make_tensor, make_tensor_value_info, ModelProto
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from furiosa_sdk_quantizer.frontend.onnx.transformer import utils
from furiosa_sdk_quantizer.frontend.onnx.quantizer.utils import (QuantizationMode, make_tensor_annotation,
                                                                 calculate_activation_quant_params,
                                                                 calculate_weight_quant_params,
                                                                 attribute_to_kwargs, append_suffix, get_input_tensors,
                                                                 is_float_tensor)
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model


class FuriosaONNXQuantizer:
    def __init__(self,
                 model: onnx.ModelProto,
                 per_channel: bool,
                 static: bool,
                 mode: QuantizationMode,
                 dynamic_ranges: Dict[str, Tuple[float, float]]):
        # copy model
        copy_model = ModelProto()
        copy_model.CopyFrom(model)
        self.model = copy_model

        self.input_tensors = get_input_tensors(model)

        # raise Exception if input_tensor is not defined in model.graph.input
        for input_tensor in self.input_tensors:
            if input_tensor not in [input.name for input in self.model.graph.input]:
                raise Exception('input_tensor: %s is not defined in model.graph.input' % input_tensor)

        # set quantization scheme
        self.per_channel = per_channel
        self.static = static  # TODO develop dynamic quantization scheme?
        self.dynamic_ranges = dynamic_ranges

        # set quantization option guided by mode
        self.quant_mode = QuantizationMode()
        self.mode = mode
        self.input_qtype = self.weight_qtype = onnx.TensorProto.INT8

        # set model a proto to use
        self.initializer = {init.name: init for init in self.model.graph.initializer}
        self.value_info = {vi.name: vi for vi in
                           list(self.model.graph.value_info)
                           + list(self.model.graph.input) + list(self.model.graph.output)}

        assert len(self.value_info) == len(list(self.model.graph.value_info)) + len(
            list(self.model.graph.input)) + len(list(self.model.graph.output))

        # (Case1) check if model is optimized: all value_infos are given.
        for node in self.model.graph.node:
            for name in list(node.input) + list(node.output):
                if name in self.initializer.keys():
                    continue

                if name not in self.value_info.keys():
                    raise Exception(
                        'value_info for %s is missing. Optimize model before quantization.' % name)

        # (Case2) raise Exception if dynamic_range is missing
        for key, vi in self.value_info.items():
            if not is_float_tensor(vi):
                continue

            if key in self.initializer.keys():
                continue

            if key not in dynamic_ranges.keys():
                raise Exception('dynamic_range for %s is missing' % key)

        # (Case3) raise Exception if dynamic_range is not defined in model.graph.value_info
        for key in dynamic_ranges:
            if key not in self.value_info.keys():
                raise Exception('dynamic range: %s is not defined in model.graph.value_info' % key)

        # stack intermediate result of quantization
        self._quant_node = {}
        self._quant_weight = {}
        self._quant_param = {}
        self._quant_value_info = {}
        self._quant_annotation = {}

        # quant model.graph field key for quantized model checker
        self._quant_initializer_key = list()
        self._quant_annotation_key = list()
        self._quant_value_info_key = list()

    def quantize(self) -> onnx.ModelProto:
        # quantize weight and activation
        self.quantize_model()

        # make quantized model and update quantized weight, scale, zero_point
        self.build_quantized_model()

        # check quantized model
        self.check_model()

        return self.model

    def quantize_model(self):
        self._quantize_activation()
        self._quantize_weight()

    def build_quantized_model(self):
        self.count = 0
        for node in self.model.graph.node:
            # TODO tmp assumption: Original model with QuantizeLinear and DequantizeLinear is not acceptable.
            if any(op == node.op_type for op in ['QuantizeLinear', 'DequantizeLinear']):
                raise Exception('Original model with %s is not acceptable.' % node.op_type)

            for idx, node_input in enumerate(node.input):
                if not is_float_tensor(self.value_info[node_input]):
                    continue
                if node_input + '_scale' not in self._quant_param.keys():
                    continue
                self.make_quant_dequant_node(node_input)
                node.input[idx] += '_dequantized'

                if node.input[idx] + '_' + str(self.count - 1) in self._quant_node:
                    node.input[idx] += '_' + str(self.count - 1)

            self._quant_node.update({node.output[0]: node})

        for output in self.model.graph.output:
            if not is_float_tensor(self.value_info[output.name]):
                continue
            if output.name + '_scale' not in self._quant_param.keys():
                continue
            self.make_quant_dequant_node(output.name)

            self.model.graph.value_info.append(output)
            output.name += '_dequantized'
            if output.name + str(self.count - 1) in self._quant_node:
                output.name += str(self.count - 1)
            self._quant_value_info.pop(output.name)

        self.model = utils.rebuild_model(model=self.model, new_nodes=self._quant_node.values(),
                                         eliminate=False)

        self._update_graph_field(field='initializer',
                                 proto=list(self._quant_param.values()) +
                                       list(self.initializer.values()))
        self._update_graph_field(field='value_info',
                                 proto=list(self._quant_value_info.values()) +
                                       list(self.model.graph.value_info))
        self._update_graph_field(field='quantization_annotation',
                                 proto=self._quant_annotation.values())

        if self.mode == QuantizationMode.dfg:
            self.model = DFGImportable(self.model).transform()
        elif self.mode == QuantizationMode.fake:
            self.model = ONNXRuntimeExecutable(self.model).transform()
        else:
            raise Exception('Unsupported mode.')

        return self.model

    def make_quant_dequant_node(self, node_input):
        # make quantizelinear node
        self._stack_quant_node(op_type='QuantizeLinear',
                               inputs=[node_input, node_input + '_scale', node_input + '_zero_point'],
                               outputs=[node_input + '_quantized']),
        self._stack_quant_vi_and_qa_helper(name=node_input, name_quant=node_input + '_quantized',
                                           elem_type=self._quant_param[node_input + '_zero_point'].data_type,
                                           quant_vi_dict=self._quant_value_info)
        # make dequantizelinear node
        output = node_input + '_dequantized'
        if output in self._quant_node.keys():
            output = node_input + '_dequantized_%d' % self.count
            self.count += 1
        self._stack_quant_node(op_type='DequantizeLinear',
                               inputs=[node_input + '_quantized', node_input + '_scale',
                                       node_input + '_zero_point'],
                               outputs=[output])
        self._stack_quant_vi_and_qa_helper(name=node_input, name_quant=output,
                                           elem_type=onnx.TensorProto.FLOAT,
                                           quant_vi_dict=self._quant_value_info)

    def check_model(self):
        check_runnable = True
        if self.mode == self.quant_mode.dfg:
            # pass runnable check, as dfg mode does not assume to run on onnxruntime
            check_runnable = False
        check_model(self.model, check_runnable)

        self._quant_value_info_key = [vi.name for vi in list(self.model.graph.value_info)
                                      + list(self.model.graph.input) + list(self.model.graph.output)]
        self._quant_initializer_key = [init.name for init in self.model.graph.initializer]
        self._quant_annotation_key = [annot.tensor_name for annot in self.model.graph.quantization_annotation]

        self._check_quant_initializer()
        self._check_quant_value_info()

        if self.mode == QuantizationMode.dfg:
            self._check_quant_annotation()
            self._check_quant_param()

    def _quantize_activation(self):
        act_quant_param = calculate_activation_quant_params(self.dynamic_ranges,
                                                            self.model.graph.node,
                                                            self.input_qtype,
                                                            self.value_info)
        suffix = ['_zero_point', '_scale']
        for name, (zp, s) in act_quant_param.items():
            zp_name, s_name = append_suffix(name=name, suffix=suffix)

            self._stack_quant_param(name_zp=zp_name, name_scale=s_name, data_type_zp=self.weight_qtype,
                                    dims=[], vals_zp=zp.flatten(), vals_scale=s.flatten())

    def _quantize_weight(self):
        disabled = True if os.environ.get('TQDM_DISABLE') else False
        for node in tqdm.tqdm(self.model.graph.node, desc='Quantization', disable=disabled):
            if any(node.op_type == op for op in ['Conv', 'ConvTranspose']):
                self._quantize_conv_weight_layer(node)
            elif any(node.op_type == op for op in ['MatMul', 'Add', 'Mul', 'Div']):
                self._quantize_matmul_weight_layer(node)
            elif node.op_type == 'Clip':
                self._quantize_clip_minmax(node)
            elif node.op_type == 'Pad':
                self._quantize_pad_constant(node)
            else:
                continue

    def _quantize_pad_constant(self, node):
        mode = attribute_to_kwargs(node.attribute)['mode']

        if mode != b'constant':
            return

        try:
            w_init = self.initializer[node.input[2]]
        except IndexError:
            name = '%s_constant_value' % node.input[0]
            node.input.append(name)

            vi = make_tensor_value_info(name=name, elem_type=onnx.TensorProto.FLOAT, shape=[])
            self.model.graph.input.append(vi)
            self.value_info.update({name: vi})

            w_init = make_tensor(name=name, data_type=onnx.TensorProto.FLOAT, dims=[], vals=[float(0)])
            self.model.graph.initializer.append(w_init)
            self.initializer.update({name: w_init})

        self._quantize_weight_per_layer(w_init)

    def _quantize_clip_minmax(self, node):
        s = numpy_helper.to_array(self._get_quant_param(node.input[0], '_scale'))
        zp = numpy_helper.to_array(self._get_quant_param(node.input[0], '_zero_point'))
        assert len(node.input) == 3

        for idx, input in enumerate(node.input):
            if input not in self.initializer.keys():
                continue

            self._stack_quant_param(name_zp=input + '_zero_point', name_scale=input + '_scale',
                                    data_type_zp=self.input_qtype,
                                    dims=[], vals_zp=[zp], vals_scale=[s])

    def _quantize_matmul_weight_layer(self, node):
        for input in node.input:
            if input not in self.initializer.keys():
                continue
            w_init = self.initializer[input]
            self._quantize_weight_per_layer(w_init)

    def _quantize_conv_weight_layer(self, node):
        try:
            w_init = self.initializer[node.input[1]]
        except KeyError:
            return

        if self.per_channel:
            self._quantize_weight_per_channel(w_init)
        else:
            self._quantize_weight_per_layer(w_init)

        if len(node.input) == 3:
            b_init = self.initializer[node.input[2]]
            i_scale = self._get_quant_param(node.input[0], '_scale')
            w_scale = self._get_quant_param(node.input[1], '_scale')
            self._quantize_bias(b_init, input_scale=i_scale, weight_scale=w_scale)

    def _quantize_weight_per_layer(self, weight_init: onnx.TensorProto) -> None:
        weight = numpy_helper.to_array(weight_init)
        zp, s = calculate_weight_quant_params(data=weight.flatten(), weight_qtype=self.weight_qtype)

        suffix = ['_quantized', '_zero_point', '_scale']
        qweight_name, zp_name, s_name = append_suffix(name=weight_init.name, suffix=suffix)

        self._stack_quant_param(name_zp=zp_name, name_scale=s_name, data_type_zp=self.weight_qtype,
                                dims=[], vals_zp=[zp], vals_scale=[s])

    def _quantize_weight_per_channel(self, weight_init: onnx.TensorProto) -> None:
        weight = numpy_helper.to_array(weight_init)
        assert weight.ndim == 4

        zp_list = []
        s_list = []

        num_output_channels = weight.shape[0]
        for i in range(num_output_channels):
            # for each channel, compute quantization data. Assuming (M x C/group x kH x kW)
            per_channel_weight = weight[i, :, :, :].flatten()
            zp, s = calculate_weight_quant_params(per_channel_weight, self.weight_qtype)
            zp_list.append(zp)
            s_list.append(s)

        suffix = ['_quantized', '_zero_point', '_scale']
        qweight_name, zp_name, s_name = append_suffix(name=weight_init.name, suffix=suffix)

        self._stack_quant_param(name_zp=zp_name, name_scale=s_name, data_type_zp=self.weight_qtype,
                                dims=[num_output_channels], vals_zp=zp_list, vals_scale=s_list)

    def _quantize_bias(self, b_init: onnx.TensorProto,
                       input_scale: onnx.TensorProto,
                       weight_scale: onnx.TensorProto) -> None:
        bias = numpy_helper.to_array(b_init)
        b_scale = numpy_helper.to_array(input_scale) * numpy_helper.to_array(weight_scale)
        # The minimum positive (subnormal) value is 2 ** -149 for IEEE 754 single-precision binary floating-point format
        # source: https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Exponent_encoding
        b_scale = np.clip(b_scale, 2 ** -149, None)

        qtype = onnx.TensorProto.INT32
        b_zero_point = np.zeros_like(b_scale).astype(TENSOR_TYPE_TO_NP_TYPE[qtype])

        suffix = ['_quantized', '_zero_point', '_scale']
        qbias_name, zp_name, s_name = append_suffix(name=b_init.name, suffix=suffix)

        self._stack_quant_param(name_zp=zp_name, name_scale=s_name, data_type_zp=qtype,
                                dims=bias.shape, vals_zp=b_zero_point, vals_scale=b_scale)

    def _stack_quant_node(self, inputs: List[str], outputs: List[str], op_type: str,
                          attributes: Optional[List[onnx.AttributeProto]] = None) -> None:

        # make quantized node proto
        attr_kwargs = {}
        if attributes:
            attr_kwargs = attribute_to_kwargs(attributes)

        quant_node = make_node(op_type,
                               inputs,
                               outputs,
                               **attr_kwargs)

        # stack quantized node
        self._quant_node.update({
            outputs[0]: quant_node
        })

    def _stack_quant_param(self, name_zp, name_scale, data_type_zp, dims, vals_zp, vals_scale):
        if name_zp in self._quant_param.keys() or name_scale in self._quant_param.keys():
            return

        # make quantization parameters initializer proto
        init_zp = make_tensor(name=name_zp, data_type=data_type_zp, dims=dims,
                              vals=vals_zp)
        init_scale = make_tensor(name=name_scale, data_type=onnx.TensorProto.FLOAT, dims=dims,
                                 vals=vals_scale)

        # stack quantization parameters
        self._quant_param.update({
            init_zp.name: init_zp,
            init_scale.name: init_scale
        })

    def _stack_quant_vi_and_qa_helper(self, name, name_quant, elem_type, quant_vi_dict):
        self._stack_quant_value_info(name=name, name_quant=name_quant,
                                     elem_type=elem_type, quant_vi_dict=quant_vi_dict)

        if elem_type == onnx.TensorProto.FLOAT:
            return

        self._stack_quant_annotation(
            *append_suffix(name=name, suffix=['_quantized', '_scale', '_zero_point'])
        )

    def _stack_quant_value_info(self, name, name_quant, elem_type, quant_vi_dict):
        if name_quant in quant_vi_dict.keys():
            return

        vi = self.value_info[name]

        # make quantized value info proto
        quant_vi = make_tensor_value_info(name=name_quant,
                                          elem_type=elem_type,
                                          shape=[v.dim_value for v in vi.type.tensor_type.shape.dim])

        # stack quantization value info
        quant_vi_dict.update({
            quant_vi.name: quant_vi
        })

    def _stack_quant_annotation(self, tensor_name, scale_name, zero_point_name):
        assert '_scale' in scale_name and '_zero_point' in zero_point_name

        if tensor_name in self._quant_annotation.keys():
            return

        # make quantization annotation
        quant_annotation = make_tensor_annotation(tensor_name=tensor_name,
                                                  zero_point_name=zero_point_name, scale_name=scale_name)

        # stack quantization annotation
        self._quant_annotation.update({
            quant_annotation.tensor_name: quant_annotation
        })

    def _update_graph_field(self, field, proto):
        self.model.graph.ClearField(field)
        getattr(self.model.graph, field).extend(proto)

    def _check_quant_initializer(self):
        for init in self.model.graph.initializer:
            init_dtype = init.data_type
            if init.name.split('_')[-1] == 'scale':
                assert init_dtype == onnx.TensorProto.FLOAT, \
                    'Wrong data type for %s.' % init.name
                assert init.float_data, 'Data should not be stored in bytes: %s' % init.name
                assert all(np.isfinite(scale) and scale > 0.0 for scale in init.float_data), \
                    f"scale tensor '{init.name}' contains an invalid element that is infinite, NaN, or nonpositive: {init.float_data}"
            elif init.name.split('_')[-1] == 'quantized' and '_'.join(
                    init.name.split('_')[-2:]) != 'fake_quantized':
                assert init_dtype in [self.weight_qtype, self.input_qtype, onnx.TensorProto.INT32], \
                    'Wrong data type for %s.' % init.name
                assert init.int32_data, 'Data should not be stored in bytes: %s' % init.name
            elif any(['_'.join(init.name.split('_')[-2:]) == word for word in
                      ['zero_point', 'quantized_min', 'quantized_max']]):
                assert init_dtype in [self.weight_qtype, self.input_qtype, onnx.TensorProto.INT32], \
                    'Wrong data type for %s.' % init.name
                assert init.int32_data, 'Data should not be stored in bytes: %s' % init.name
            elif '_'.join(init.name.split('_')[-2:]) == 'fake_quantized':
                assert init_dtype == onnx.TensorProto.FLOAT, 'Wrong data type for %s' % init.name
            else:
                assert init_dtype == onnx.TensorProto.INT64 or init_dtype == onnx.TensorProto.FLOAT, \
                    'Wrong data type for %s.' % init.name

    def _check_quant_value_info(self):
        quant_inputs = [node_input for node in self.model.graph.node
                        for node_input in node.input
                        if node_input not in self._quant_initializer_key]
        quant_outputs = [node_output for node in self.model.graph.node
                         for node_output in node.output]

        # check if every node.input/output has graph.value_info
        for name in set(quant_inputs + quant_outputs):
            if name not in self._quant_value_info_key:
                raise KeyError('%s is not defined in graph.value_info' % name)

        # check if graph.value_info and graph.input/output are disjoint
        for vi in self.model.graph.value_info:
            for inp in self.model.graph.input:
                if vi.name == inp.name:
                    raise Exception('%s in graph.value_info is also defined in graph.input' % vi.name)

            for oup in self.model.graph.output:
                if vi.name == oup.name:
                    raise Exception('%s in graph.value_info is also defined in graph.output' % vi.name)

    def _check_quant_annotation(self):
        quant_inputs = [node_input for node in self.model.graph.node
                        for node_input in node.input
                        if node_input.split('_')[-1] == 'quantized']
        quant_outputs = [node_output for node in self.model.graph.node
                         for node_output in node.output
                         if node_output.split('_')[-1] == 'quantized']

        for tensor_name in set(quant_inputs + quant_outputs):
            if tensor_name not in self._quant_annotation_key:
                raise KeyError('%s is not defined in graph.quantization_annotation' % tensor_name)

    def _check_quant_param(self):
        for annotation in self.model.graph.quantization_annotation:
            for quant_param in annotation.quant_parameter_tensor_names:
                if quant_param.value not in self._quant_initializer_key:
                    raise KeyError('%s is not defined in graph.initializer' % quant_param.value)

        for init in self.model.graph.initializer:
            if init.name.split('_')[-1] == 'scale':
                if all(v == 0. for v in init.float_data):
                    assert 'quantization scale parameter should not be zero: %s' % init.name

        # check if conv bias scale is correct
        for node in self.model.graph.node:
            if node.op_type != 'QLinearConv':
                continue

            if len(node.input) != 9:
                continue

            i_scale_arr = numpy_helper.to_array(self._quant_param[node.input[1]])
            w_scale_arr = numpy_helper.to_array(self._quant_param[node.input[4]])
            b_scale_name = node.input[-1].split('_quantized')[0] + '_scale'
            b_scale_arr = numpy_helper.to_array(self._quant_param[b_scale_name])

            assert np.allclose(b_scale_arr, i_scale_arr * w_scale_arr), f'Conv bias scale is incorrect: {b_scale_name}'

    def _get_quant_param(self, origin, postfix=None):
        result = self._quant_param.get(f'{origin}{postfix or ""}', None)
        if result is None:
            raise Exception(f"dynamic-range '{origin}' is missing")
        return result


class DFGImportable:
    def __init__(self, model):
        copy_model = ModelProto()
        copy_model.CopyFrom(model)
        self.model = copy_model

        self.node = {node.name: node for node in self.model.graph.node}
        self.node_by_output = {node_output: node for node in self.model.graph.node for node_output in
                               node.output}
        self.node_by_input = {node_input: node for node in self.model.graph.node for node_input in node.input}
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
            if node.input[0] not in self.initializer.keys():
                new_nodes.append(node)
                continue

            # node.input[0] to be removed from model.graph.input
            self.graph_input.pop(node.input[0])

            # node.input[0] to be removed from model.graph.initializer
            init = self.initializer.pop(node.input[0])

            # quantize initializer
            s = self.initializer[node.input[1]]
            zp = self.initializer[node.input[2]]

            quantized_data = self._quantize_data(init, s, zp)

            # node.output[0] to be updated to model.graph.initializer instead
            self.initializer.update({node.output[0]: make_tensor(node.output[0], zp.data_type,
                                                                 init.dims,
                                                                 quantized_data.flatten())})

            # node.output[0] to be removed from model.graph.value_info
            vi = self.value_info.pop(node.output[0])

            # node.output[0] to be updated to model.graph.input instead
            self.graph_input.update({node.output[0]: vi})

            rm_nodes.append(node)

        self.model = utils.rebuild_model(model=self.model,
                                         new_nodes=[node for node in new_nodes if node not in rm_nodes],
                                         eliminate=False,
                                         renaming=False)

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

            self.value_info.pop(node_i0.output[0])
            self.value_info.pop(node_i1.output[0])
            self.value_info.pop(node_o0.input[0])

            node_i2 = None
            if len(node.input) == 3:
                node_i2 = self.node_by_output[node.input[2]]
                rm_nodes.append(node_i2)

                self.value_info.pop(node_i2.output[0])

            rm_nodes.extend([node])
            new_nodes.append(self._make_integer_arithmetic_operator(node, node_i0, node_i1, node_o0, node_i2))

        self.model = utils.rebuild_model(model=self.model,
                                         new_nodes=[node for node in new_nodes if node not in rm_nodes],
                                         eliminate=False)

        self._update_graph_field(field='value_info', proto=self.value_info.values())

        check_model(self.model, check_runnable=False)

    @staticmethod
    def _make_integer_arithmetic_operator(node, node_i0, node_i1, node_o0, node_i2=None):
        quant_op_type = 'QLinear%s' % node.op_type

        quant_inputs = [*node_i0.input, *node_i1.input, *node_o0.input[1:]]

        if node_i2:
            quant_inputs.append(node_i2.input[0])

        quant_outputs = [node_o0.output[0]]

        attributes = node.attribute
        attr_kwargs = {}
        if attributes:
            attr_kwargs = attribute_to_kwargs(attributes)

        quant_node = make_node(quant_op_type,
                               quant_inputs,
                               quant_outputs,
                               **attr_kwargs)

        return quant_node

    @staticmethod
    def _quantize_data(data: onnx.TensorProto, scale: onnx.TensorProto,
                       zero_point: onnx.TensorProto) -> np.array:
        data_arr = np.atleast_1d(numpy_helper.to_array(data)).astype(np.float32)
        scale_arr = np.atleast_1d(numpy_helper.to_array(scale)).astype(np.float32)
        zero_point_arr = np.atleast_1d(numpy_helper.to_array(zero_point)).astype(np.float32)

        if data_arr.ndim != scale_arr.ndim:
            # assume output channel quantization
            scale_arr = scale_arr.reshape((-1,) + (1,) * (data_arr.ndim - 1))

        if data_arr.ndim != zero_point_arr.ndim:
            # assume output channel quantization
            zero_point_arr = zero_point_arr.reshape((-1,) + (1,) * (data_arr.ndim - 1))

        quantized_data = np.round(data_arr / scale_arr) + zero_point_arr

        np_dtype_info = np.iinfo(TENSOR_TYPE_TO_NP_TYPE[zero_point.data_type])
        np_dtype_info_min = np_dtype_info.min
        np_dtype_info_max = np_dtype_info.max

        return np.clip(quantized_data,
                       np_dtype_info_min,
                       np_dtype_info_max).astype(TENSOR_TYPE_TO_NP_TYPE[zero_point.data_type])

    def _update_graph_field(self, field, proto):
        self.model.graph.ClearField(field)
        getattr(self.model.graph, field).extend(proto)


class ONNXRuntimeExecutable(DFGImportable):
    def __init__(self, model):
        super(ONNXRuntimeExecutable, self).__init__(model)

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

            if node.input[0] not in self.initializer.keys():
                new_nodes.append(node)
                continue

            rm_nodes.append(node)
            rm_nodes.extend([dequant_node for dequant_node in self.model.graph.node
                             if dequant_node.op_type == 'DequantizeLinear'
                             if dequant_node.input[0] == node.output[0]])

        for node in self.model.graph.node:
            if node.op_type == 'QuantizeLinear' or node.op_type == 'DequantizeLinear':
                continue

            for idx, node_input in enumerate(node.input):
                if '_dequantized' not in node_input:
                    continue

                init_name = node_input.split('_dequantized')[0]
                if init_name not in self.initializer.keys():
                    continue

                node.input[idx] = init_name + '_fake_quantized'

                init = self.initializer[init_name]
                s = self.initializer[init_name + '_scale']
                zp = self.initializer[init_name + '_zero_point']

                fake_quantized_data = self._fake_quantize_data(init, s, zp)

                self.initializer.update({
                    init_name + '_fake_quantized': make_tensor(name=init_name + '_fake_quantized',
                                                               data_type=onnx.TensorProto.FLOAT,
                                                               dims=init.dims,
                                                               vals=fake_quantized_data.flatten())
                })

        self.model = utils.rebuild_model(model=self.model,
                                         new_nodes=[node for node in new_nodes if node not in rm_nodes],
                                         eliminate=True)
        self._update_graph_field(field='initializer', proto=self.initializer.values())

        check_model(self.model, check_runnable=True)

    @staticmethod
    def _dequantize_data(data: onnx.TensorProto, scale: onnx.TensorProto,
                         zero_point: onnx.TensorProto) -> np.array:
        data_arr = np.atleast_1d(numpy_helper.to_array(data)).astype(np.float32)
        scale_arr = np.atleast_1d(numpy_helper.to_array(scale)).astype(np.float32)
        zero_point_arr = np.atleast_1d(numpy_helper.to_array(zero_point)).astype(np.float32)

        if data_arr.ndim != scale_arr.ndim:
            # assume output channel quantization
            scale_arr = scale_arr.reshape((-1,) + (1,) * (data_arr.ndim - 1))

        if data_arr.ndim != zero_point_arr.ndim:
            # assume output channel quantization
            zero_point_arr = zero_point_arr.reshape((-1,) + (1,) * (data_arr.ndim - 1))

        return (data_arr - zero_point_arr) * scale_arr

    def _fake_quantize_data(self, data: onnx.TensorProto, scale: onnx.TensorProto,
                            zero_point: onnx.TensorProto) -> np.array:

        quantized_data = self._quantize_data(data, scale, zero_point)

        dequantized_data = self._dequantize_data(
            make_tensor(name=data.name, data_type=zero_point.data_type, dims=data.dims,
                        vals=quantized_data.flatten()),
            scale, zero_point)

        return dequantized_data
