from typing import Callable, Dict, List, Optional

import numpy as np
import onnx

__DOMAIN__ = ''
__OPSET_VERSION__ = 13

from furiosa.optimizer.frontend.onnx.transformer.convert_negative_pads_to_slice import (
    ConvertNegativePadsToSlice,
)
from furiosa.optimizer.frontend.onnx.transformer.convert_prelu_to_relu import ConvertPReluToRelu
from furiosa.optimizer.frontend.onnx.transformer.fuse_batchnorm import FuseBatchNorm
from furiosa.optimizer.frontend.onnx.transformer.fuse_gather_matmul import FuseGatherMatMul
from furiosa.optimizer.frontend.onnx.transformer.polish_model import PolishModel
from furiosa.optimizer.frontend.onnx.utils.version_checker import CheckVersion


def optimize_model(
    model: onnx.ModelProto,  # pylint: disable=no-member
    input_shapes: Optional[Dict[str, List[int]]] = None,
    opset_version: int = __OPSET_VERSION__,
) -> onnx.ModelProto:  # pylint: disable=no-member
    model = _transform([CheckVersion(opset_version).transform], model)
    model = _transform([PolishModel(input_shapes).transform], model)

    # TODO check if graph_transform should apply.
    model = _transform([_reify], model)
    return model


def _transform(
    transformers: List[Callable[[onnx.ModelProto], onnx.ModelProto]],  # pylint: disable=no-member
    model: onnx.ModelProto,  # pylint: disable=no-member
) -> onnx.ModelProto:  # pylint: disable=no-member
    for transform in transformers:
        model = transform(model)
    return model


def _reify(model: onnx.ModelProto) -> onnx.ModelProto:  # pylint: disable=no-member
    transformers = [
        ConvertNegativePadsToSlice().transform,
        FuseBatchNorm().transform,
        FuseGatherMatMul().transform,
        ConvertPReluToRelu().transform,
    ]
    return _transform(transformers, model)
