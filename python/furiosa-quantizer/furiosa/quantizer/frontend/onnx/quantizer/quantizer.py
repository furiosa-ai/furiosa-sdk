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

from collections import defaultdict
import copy
import itertools
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.helper import ModelProto, make_node, make_tensor, make_tensor_value_info
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import tqdm

from furiosa.quantizer.frontend.onnx.quantizer import eliminate_clipper, quantizer_mode
from furiosa.quantizer.frontend.onnx.quantizer.utils import (
    QuantizationMode,
    append_suffix,
    calculate_activation_quant_params,
    calculate_weight_quant_params,
    get_input_tensors,
    is_float_tensor,
)
from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model

transformers = [
    # eliminate clippers like Relu, Clip into Conv, Add
    # and replace the letters' output quantization parameters with the formers
    # for marginal accuracy gain
    eliminate_clipper.EliminateClipper().transform
]


class FuriosaONNXQuantizer:
    def __init__(
        self,
        model: onnx.ModelProto,
        per_channel: bool,
        static: bool,
        mode: QuantizationMode,
        dynamic_ranges: Dict[str, Tuple[float, float]],
        raw_data=True,
    ):
        """
        - raw_data:
                if True, quantized weight/bias/scale/zero_point/etc.. will be stored in bytes.
                Else, in data_type
        """
        # copy model
        copy_model = ModelProto()
        copy_model.CopyFrom(model)
        self.model = copy_model

        self.raw_data = raw_data

        self.input_tensors = list(map(lambda tensor: tensor[0], get_input_tensors(model)))

        # raise Exception if input_tensor is not defined in model.graph.input
        for input_tensor in self.input_tensors:
            if input_tensor not in [input.name for input in self.model.graph.input]:
                raise Exception(f'input_tensor: {input_tensor} is not defined in model.graph.input')

        # set quantization scheme
        self.per_channel = per_channel
        self.static = static  # TODO develop dynamic quantization scheme?
        self.dynamic_ranges = dynamic_ranges

        # set quantization option guided by mode
        self.mode = mode
        self.activation_qtype = self.weight_qtype = onnx.TensorProto.INT8

        # use uint8 dtype for activation in fake_quant mode
        if mode == QuantizationMode.FAKE:
            self.activation_qtype = onnx.TensorProto.UINT8

        # set model a proto to use
        self.initializer = {init.name: init for init in self.model.graph.initializer}
        self.consumer_map = defaultdict(list)
        for node in model.graph.node:
            for tensor in node.input:
                self.consumer_map[tensor].append(node)
        self.init_vi = [
            make_tensor_value_info(init.name, init.data_type, init.dims)
            for init in self.model.graph.initializer
        ]
        self.value_info_all = {
            vi.name: vi
            for vi in itertools.chain(
                self.model.graph.value_info,
                self.model.graph.input,
                self.model.graph.output,
                self.init_vi,
            )
        }

        # (Case1) check if model is optimized: all value_infos are given with valid dimension/type info.
        utils.check_value_info(self.model)

        # (Case2) raise Exception if dynamic_range is missing
        for key, vi in self.value_info_all.items():
            if not is_float_tensor(vi):
                continue

            if key in self.initializer:
                continue

            if key not in dynamic_ranges:
                raise Exception(f'dynamic_range for {key} is missing')

        # (Case3) raise Exception if dynamic_range is not defined in model.graph.value_info
        for key in dynamic_ranges:
            if key not in self.value_info_all:
                raise Exception(f'dynamic range: {key} is not defined in model.graph.value_info')

        # stack intermediate result of quantization
        self._quant_node = {}
        self._quant_weight = {}
        self._quant_param = {}
        self._quant_value_info = {}

        # quant model.graph field key for quantized model checker
        self._quant_initializer_key = []
        self._quant_value_info_key = []

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
        self.model = self.make_intermediate_representation()

        for transform in transformers:
            self.model = transform(self.model)

        if self.mode == QuantizationMode.DFG:
            self.model = quantizer_mode.DFGImportable(self.model, self.raw_data).transform()
        elif self.mode == QuantizationMode.FAKE:
            self.model = quantizer_mode.ONNXRuntimeExecutable(self.model, self.raw_data).transform()

        return self.model

    def make_intermediate_representation(self):
        for node in self.model.graph.node:
            # TODO tmp assumption: Original model with QuantizeLinear and DequantizeLinear is not acceptable.
            if any(op == node.op_type for op in ['QuantizeLinear', 'DequantizeLinear']):
                raise Exception(f'Original model with {node.op_type} is not acceptable.')

            # uses orig_node as an index getter
            # to avoid updating every consumer_map that contains the mutated node
            orig_node = copy.deepcopy(node)
            for idx, node_input in enumerate(node.input):
                if not is_float_tensor(self.value_info_all[node_input]):
                    continue
                if node_input + '_scale' not in self._quant_param:
                    continue

                suffix = (
                    self.consumer_map[node_input].index(orig_node)
                    if len(self.consumer_map[node_input]) > 1
                    else None
                )
                self.make_quant_dequant_node(node_input, suffix)
                node.input[idx] += '_dequantized' + (
                    f'_{str(suffix)}' if suffix is not None else ''
                )

            self._quant_node.update({node.output[0]: node})
        for output in self.model.graph.output:
            if not is_float_tensor(self.value_info_all[output.name]):
                continue
            if output.name + '_scale' not in self._quant_param:
                continue
            self.make_quant_dequant_node(output.name)

            self.model.graph.value_info.append(output)
            output.name += '_dequantized'
            self._quant_value_info.pop(output.name)

        self.model = utils.rebuild_model(
            model=self.model, new_nodes=list(self._quant_node.values()), eliminate=False
        )

        self._update_graph_field(
            field='initializer',
            proto=list(self._quant_param.values()) + list(self.initializer.values()),
        )
        self._update_graph_field(
            field='value_info',
            proto=list(self._quant_value_info.values()) + list(self.model.graph.value_info),
        )

        return self.model

    def make_quant_dequant_node(self, node_input, idx=None):
        scale = node_input + '_scale'
        zero_point = node_input + '_zero_point'
        qlinear_output = node_input + '_quantized'
        dqlinear_output = node_input + '_dequantized'
        if idx is not None:
            dqlinear_output += f'_{str(idx)}'

        # make quantizelinear node
        self._stack_quant_node(
            op_type='QuantizeLinear',
            inputs=[node_input, scale, zero_point],
            outputs=[qlinear_output],
        )
        self._stack_quant_vi_and_qa_helper(
            name=node_input,
            name_quant=qlinear_output,
            elem_type=self._quant_param[zero_point].data_type,
            quant_vi_dict=self._quant_value_info,
        )

        # make dequantizelinear node
        self._stack_quant_node(
            op_type='DequantizeLinear',
            inputs=[qlinear_output, scale, zero_point],
            outputs=[dqlinear_output],
        )
        self._stack_quant_vi_and_qa_helper(
            name=node_input,
            name_quant=dqlinear_output,
            elem_type=onnx.TensorProto.FLOAT,
            quant_vi_dict=self._quant_value_info,
        )

    def check_model(self):
        check_runnable = True
        if self.mode == QuantizationMode.DFG:
            # pass runnable check, as dfg mode does not assume to run on onnxruntime
            check_runnable = False
        check_model(self.model, check_runnable)

        self._quant_value_info_key = [
            vi.name
            for vi in list(self.model.graph.value_info)
            + list(self.model.graph.input)
            + list(self.model.graph.output)
        ]
        self._quant_initializer_key = [init.name for init in self.model.graph.initializer]

        self._check_quant_initializer()
        self._check_quant_value_info()

        if self.mode == QuantizationMode.DFG:
            self._check_quant_param()

    def _quantize_activation(self):
        act_quant_param = calculate_activation_quant_params(
            self.dynamic_ranges, self.model.graph.node, self.activation_qtype, self.value_info_all
        )
        suffix = ['_zero_point', '_scale']
        for name, (zp, s) in act_quant_param.items():
            zp_name, s_name = append_suffix(name=name, suffix=suffix)

            self._stack_quant_param(
                name_zp=zp_name,
                name_scale=s_name,
                data_type_zp=self.activation_qtype,
                dims=zp.shape,
                vals_zp=zp,
                vals_scale=s,
            )

    def _quantize_weight(self):
        disabled = bool(os.environ.get('TQDM_DISABLE'))
        for node in tqdm.tqdm(
            self.model.graph.node, desc='Quantization', disable=disabled, unit='operators'
        ):
            if node.op_type == 'Conv':
                self._quantize_conv_weight(node, output_channel_axis=0)
            elif node.op_type == 'ConvTranspose':
                self._quantize_conv_weight(node, output_channel_axis=1)
            elif any(
                node.op_type == op for op in ['MatMul', 'Add', 'Mul', 'Div', 'Sub', 'Concat', 'Pow']
            ):
                self._quantize_matmul_weight(node)
            elif node.op_type == 'Clip':
                self._quantize_clip_minmax(node)
            elif node.op_type == 'Pad':
                self._quantize_pad_constant(node)
            else:
                continue

    def _quantize_pad_constant(self, node):
        mode = utils.get_attribute(node.attribute, "mode", b"constant")

        if mode != b'constant':
            return

        try:
            w_init = self.initializer[node.input[2]]
        except IndexError:
            name = f'{node.input[0]}_constant_value'
            node.input.append(name)

            vi = make_tensor_value_info(name=name, elem_type=onnx.TensorProto.FLOAT, shape=[])
            self.model.graph.input.append(vi)
            self.value_info_all.update({name: vi})

            if not self.raw_data:
                w_init = make_tensor(
                    name=name, data_type=onnx.TensorProto.FLOAT, dims=[], vals=[float(0)]
                )
            else:
                w_init = numpy_helper.from_array(np.array(0.0), name=name)

            self.model.graph.initializer.append(w_init)
            self.initializer.update({name: w_init})

        self._quantize_weight_per_layer(w_init)

    def _quantize_clip_minmax(self, node):
        s = numpy_helper.to_array(self._get_quant_param(node.input[0], '_scale'))
        zp = numpy_helper.to_array(self._get_quant_param(node.input[0], '_zero_point'))

        for tensor_name in node.input:
            if tensor_name not in self.initializer:
                continue

            self._stack_quant_param(
                name_zp=tensor_name + '_zero_point',
                name_scale=tensor_name + '_scale',
                data_type_zp=self.activation_qtype,
                dims=zp.shape,
                vals_zp=zp,
                vals_scale=s,
            )

    def _quantize_matmul_weight(self, node):
        for tensor_name in node.input:
            if tensor_name not in self.initializer:
                continue
            w_init = self.initializer[tensor_name]
            self._quantize_weight_per_layer(w_init)

    def _quantize_conv_weight(self, node, output_channel_axis):
        try:
            w_init = self.initializer[node.input[1]]
        except KeyError:
            return

        if self.per_channel:
            self._quantize_weight_per_axis(w_init, axis=output_channel_axis)
        else:
            self._quantize_weight_per_layer(w_init)

        if len(node.input) == 3:
            b_init = self.initializer[node.input[2]]
            i_scale = self._get_quant_param(node.input[0], '_scale')
            w_scale = self._get_quant_param(node.input[1], '_scale')
            self._quantize_bias(b_init, input_scale=i_scale, weight_scale=w_scale)

    def _quantize_weight_per_layer(self, weight_init: onnx.TensorProto) -> None:
        weight = numpy_helper.to_array(weight_init)
        zp, s = calculate_weight_quant_params(
            data=weight.flatten(), weight_qtype=self.weight_qtype, name=weight_init.name
        )

        suffix = ['_quantized', '_zero_point', '_scale']
        _, zp_name, s_name = append_suffix(name=weight_init.name, suffix=suffix)

        self._stack_quant_param(
            name_zp=zp_name,
            name_scale=s_name,
            data_type_zp=self.weight_qtype,
            dims=[],
            vals_zp=np.asarray(zp, dtype=TENSOR_TYPE_TO_NP_TYPE[self.weight_qtype]),
            vals_scale=np.asarray(s, dtype=np.float32),
        )

    def _quantize_weight_per_axis(self, weight_init: onnx.TensorProto, axis: int) -> None:
        assert len(weight_init.dims) == 4, f'weight should have rank 4: {repr(weight_init)}'
        weight = numpy_helper.to_array(weight_init)

        num_output_channels = weight.shape[axis]
        s_list = []
        zp_list = []
        for i in range(num_output_channels):
            indices = [slice(None)] * weight.ndim
            indices[axis] = i
            per_channel_weight = weight[tuple(indices)].flatten()
            zp, s = calculate_weight_quant_params(
                per_channel_weight, self.weight_qtype, weight_init.name
            )
            s_list.append(s)
            zp_list.append(zp)

        suffix = ['_quantized', '_zero_point', '_scale']
        _, zp_name, s_name = append_suffix(name=weight_init.name, suffix=suffix)

        self._stack_quant_param(
            name_zp=zp_name,
            name_scale=s_name,
            data_type_zp=self.weight_qtype,
            dims=[num_output_channels],
            vals_zp=np.asarray(zp_list, dtype=TENSOR_TYPE_TO_NP_TYPE[self.weight_qtype]),
            vals_scale=np.asarray(s_list, dtype=np.float32),
        )

    def _quantize_bias(
        self,
        b_init: onnx.TensorProto,
        input_scale: onnx.TensorProto,
        weight_scale: onnx.TensorProto,
    ) -> None:
        b_scale = numpy_helper.to_array(input_scale) * numpy_helper.to_array(weight_scale)
        b_scale = np.where(b_scale == 0.0, np.float32(2**-149), b_scale)

        qtype = onnx.TensorProto.INT32
        b_zero_point = np.zeros_like(b_scale).astype(TENSOR_TYPE_TO_NP_TYPE[qtype])

        suffix = ['_quantized', '_zero_point', '_scale']
        _, zp_name, s_name = append_suffix(name=b_init.name, suffix=suffix)

        self._stack_quant_param(
            name_zp=zp_name,
            name_scale=s_name,
            data_type_zp=qtype,
            dims=b_zero_point.shape,
            vals_zp=b_zero_point,
            vals_scale=b_scale,
        )

    def _stack_quant_node(
        self,
        inputs: List[str],
        outputs: List[str],
        op_type: str,
        attributes: Optional[List[onnx.AttributeProto]] = None,
    ) -> None:
        if attributes is None:
            attributes = []

        # make quantized node proto
        attr_kwargs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in attributes}

        quant_node = make_node(op_type, inputs, outputs, **attr_kwargs)

        # stack quantized node
        self._quant_node.update({outputs[0]: quant_node})

    def _stack_quant_param(self, name_zp, name_scale, data_type_zp, dims, vals_zp, vals_scale):
        if name_zp in self._quant_param or name_scale in self._quant_param:
            return
        # make quantization parameters initializer proto
        if self.raw_data:
            init_zp = make_tensor(
                name=name_zp, data_type=data_type_zp, dims=dims, vals=vals_zp.tobytes(), raw=True
            )
            init_scale = make_tensor(
                name=name_scale,
                data_type=onnx.TensorProto.FLOAT,
                dims=dims,
                vals=vals_scale.tobytes(),
                raw=True,
            )
        else:
            # Converts numpy 0d array to list of it,
            # for onnx.helper.make_tensor requires vals to be a list.
            vals_zp = np.atleast_1d(vals_zp)
            vals_scale = np.atleast_1d(vals_scale)
            init_zp = make_tensor(name=name_zp, data_type=data_type_zp, dims=dims, vals=vals_zp)
            init_scale = make_tensor(
                name=name_scale, data_type=onnx.TensorProto.FLOAT, dims=dims, vals=vals_scale
            )

        # stack quantization parameters
        self._quant_param.update({init_zp.name: init_zp, init_scale.name: init_scale})

    def _stack_quant_vi_and_qa_helper(self, name, name_quant, elem_type, quant_vi_dict):
        self._stack_quant_value_info(
            name=name, name_quant=name_quant, elem_type=elem_type, quant_vi_dict=quant_vi_dict
        )

        if elem_type == onnx.TensorProto.FLOAT:
            return

    def _stack_quant_value_info(self, name, name_quant, elem_type, quant_vi_dict):
        if name_quant in quant_vi_dict:
            return

        vi = self.value_info_all[name]

        # make quantized value info proto
        quant_vi = make_tensor_value_info(
            name=name_quant,
            elem_type=elem_type,
            shape=[v.dim_value for v in vi.type.tensor_type.shape.dim],
        )

        # stack quantization value info
        quant_vi_dict.update({quant_vi.name: quant_vi})

    def _update_graph_field(self, field, proto):
        self.model.graph.ClearField(field)
        getattr(self.model.graph, field).extend(proto)

    def _check_quant_initializer(self):
        for init in self.model.graph.initializer:
            init_dtype = init.data_type
            # check quant params' data type and field
            if init.name.split('_')[-1] == 'scale':
                assert (
                    init_dtype == onnx.TensorProto.FLOAT
                ), f'Wrong data type {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init_dtype]} for {init.name}'
                if not self.raw_data:
                    assert (
                        init.float_data and not init.raw_data
                    ), f'the scale {init.name} should be stored in float_data and not in raw_data: {init}'
                else:
                    assert (
                        init.raw_data and not init.float_data
                    ), f'the scale {init.name} should be stored in raw_data and not in float_data: {init}'

            elif (
                init.name.split('_')[-1] == 'quantized'
                and '_'.join(init.name.split('_')[-2:]) != 'fake_quantized'
            ):
                assert init_dtype in [
                    self.weight_qtype,
                    self.activation_qtype,
                    onnx.TensorProto.INT32,
                ], f'Wrong data type {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init_dtype]} for {init.name}'

                if not self.raw_data:
                    assert (
                        init.int32_data and not init.raw_data
                    ), f'the quatized weight/bias {init.name} should be stored in int32_data and not in raw_data: {init}'
                else:
                    assert (
                        init.raw_data and not init.int32_data
                    ), f'the quatized weight/bias {init.name} should be stored in raw_data and not in int32_data: {init}'
            elif any(
                '_'.join(init.name.split('_')[-2:]) == word
                for word in ('zero_point', 'quantized_min', 'quantized_max')
            ):
                assert init_dtype in [
                    self.weight_qtype,
                    self.activation_qtype,
                    onnx.TensorProto.INT32,
                ], f'Wrong data type {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init_dtype]} for {init.name}'

                if not self.raw_data:
                    assert (
                        init.int32_data and not init.raw_data
                    ), f'the quatized weight/bias {init.name} should be stored in int32_data and not in in raw_data: {init}'
                else:
                    assert (
                        init.raw_data and not init.int32_data
                    ), f'the quatized weight/bias {init.name} should be stored in raw_data and not in int32_data: {init}'
            elif '_'.join(init.name.split('_')[-2:]) == 'fake_quantized':
                assert (
                    init_dtype == onnx.TensorProto.FLOAT
                ), f'Wrong data type {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init_dtype]} for {init.name}'
            else:
                assert init_dtype in (
                    onnx.TensorProto.INT64,
                    onnx.TensorProto.FLOAT,
                ), f'Unknown data type {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init_dtype]} for {init.name}'

        # Checks if scale/zero-point of DequantizedLienar/QuantizeLinear (OpSet < 13) are scalars.
        opset = next((opset for opset in self.model.opset_import if not opset.domain), None)
        if opset is not None and opset.version < 13:
            for node in self.model.graph.node:
                if node.op_type == "DequantizeLinear":
                    # Bypasses the checker if node.input[0] in DequantizeLinear is
                    # defined in model.graph.initializer.
                    # Since it might have 1-d scale and zero-point if per-channel quantized,
                    # which conflicts with DequantizeLinear(opset-12) spec
                    # but also is unavoidable according to our graph representations.
                    if node.input[0] in self._quant_initializer_key:
                        continue
                    scale = self._quant_param[node.input[1]]
                    zero_point = self._quant_param[node.input[2]]
                    assert (
                        not scale.dims
                    ), f"the 'x_scale' input of DequantizeLinear (OpSet {opset.version}) must be a scalar"
                    assert (
                        not zero_point.dims
                    ), f"the 'x_zero_point' input of DequantizeLinear (OpSet {opset.version}) must be a scalar"
                elif node.op_type == "QuantizeLinear":
                    scale = self._quant_param[node.input[1]]
                    zero_point = self._quant_param[node.input[2]]
                    assert (
                        not scale.dims
                    ), f"the 'y_scale' input of QuantizeLinear (OpSet {opset.version}) must be a scalar"
                    assert (
                        not zero_point.dims
                    ), f"the 'y_zero_point' input of QuantizeLinear (OpSet {opset.version}) must be a scalar"

        # Checks if quantization parameters are at best 1-d array.
        for init in self.model.graph.initializer:
            postfix = init.name.rsplit('_', maxsplit=1)[-1]
            if postfix not in ['scale', 'zero_point']:
                continue
            quant_param = self._quant_param[init.name]
            rank = len(quant_param.dims)
            assert rank <= 1, f"{init.name} has rank {rank}. {postfix} cannot have rank > 1."

    def _check_quant_value_info(self):
        quant_inputs = [
            node_input
            for node in self.model.graph.node
            for node_input in node.input
            if node_input not in self._quant_initializer_key
        ]
        quant_outputs = [
            node_output for node in self.model.graph.node for node_output in node.output
        ]

        # check if every node.input/output has graph.value_info
        for name in set(quant_inputs + quant_outputs):
            if name not in self._quant_value_info_key:
                raise KeyError(f'{name} is not defined in graph.value_info')

        # check if graph.value_info and graph.input/output are disjoint
        for vi in self.model.graph.value_info:
            for inp in self.model.graph.input:
                if vi.name == inp.name:
                    raise Exception(f'{vi.name} in graph.value_info is also defined in graph.input')

            for oup in self.model.graph.output:
                if vi.name == oup.name:
                    raise Exception(
                        f'{vi.name} in graph.value_info is also defined in graph.output'
                    )

    def _check_quant_param(self):
        # ensure that all scales are non-zero.
        for init in self.model.graph.initializer:
            if init.name.split('_')[-1] == 'scale':
                init_array = onnx.numpy_helper.to_array(init)
                assert all(
                    v != 0.0 for v in np.nditer(init_array)
                ), f"the scale '{init.name}' should be non-zero: {init_array}"

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

            assert np.allclose(
                b_scale_arr, (i_scale_arr * w_scale_arr).reshape(-1)
            ), f'Conv bias scale is incorrect: {b_scale_name}'

        # check if transposeconv bias scale is correct
        for node in self.model.graph.node:
            if node.op_type != "ConvTranspose":
                continue

            if len(node.input) != 3:
                continue

            i_scale_arr = numpy_helper.to_array(
                self._quant_param[node.input[0].split('_dequantized')[0] + '_scale']
            )
            w_scale_arr = numpy_helper.to_array(
                self._quant_param[node.input[1].split('_dequantized')[0] + '_scale']
            )
            b_scale_name = node.input[2].split('_dequantized')[0] + '_scale'
            b_scale_arr = numpy_helper.to_array(self._quant_param[b_scale_name])

            assert np.allclose(
                b_scale_arr, (i_scale_arr * w_scale_arr).reshape(-1)
            ), f'Conv bias scale is incorrect: {b_scale_name}'

    def _get_quant_param(self, origin, postfix=None):
        result = self._quant_param.get(f'{origin}{postfix or ""}', None)
        if result is None:
            raise Exception(f"dynamic-range '{origin}' is missing")
        return result
