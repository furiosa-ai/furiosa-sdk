import onnx

import onnxoptimizer as optimizer
from onnx import numpy_helper
from onnx.helper import make_tensor_value_info, make_model, make_tensor

from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer import utils


class ExtractConstantToInitializer(Transformer):
    """
    from: ArgMax -> graph output
    to: graph output
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for init in model.graph.initializer:
            if init.name not in [inp.name for inp in model.graph.input]:
                dims = numpy_helper.to_array(init).shape
                model.graph.input.append(
                    make_tensor_value_info(name=init.name, elem_type=init.data_type, shape=dims)
                )

        model = optimizer.optimize(model, passes=["extract_constant_to_initializer"])

        # make scalar initializer's dim=0 in Add, Sub, Mul, Div
        initializer = {init.name: init for init in model.graph.initializer}
        input_vi = {inp.name: inp for inp in model.graph.input}

        for node in model.graph.node:
            if not any(node.op_type == op for op in ["Add", "Sub", "Mul", "Div"]):
                continue
            for node_input in node.input:
                if node_input not in initializer:
                    continue

                init = initializer[node_input]
                vi = input_vi[node_input]
                arr = numpy_helper.to_array(init)

                if arr.size == 1 and arr.shape == (1,):
                    model.graph.initializer.remove(init)
                    model.graph.initializer.append(
                        make_tensor(
                            name=init.name,
                            data_type=init.data_type,
                            dims=(),
                            vals=numpy_helper.to_array(init),
                        )
                    )
                    model.graph.input.remove(vi)
                    model.graph.input.append(
                        make_tensor_value_info(
                            name=init.name, elem_type=vi.type.tensor_type.elem_type, shape=()
                        )
                    )

        model = make_model(model.graph)
        model = utils.rebuild_model(model, model.graph.node)

        return model
