from collections import defaultdict
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx

__DOMAIN__ = ''
__OPSET_VERSION__ = 13

from furiosa.optimizer.frontend.onnx import calibrate
from furiosa.optimizer.frontend.onnx.transformer.convert_conv1d_to_conv2d import (
    ConvertConv1dToConv2d,
)
from furiosa.optimizer.frontend.onnx.transformer.convert_negative_pads_to_slice import (
    ConvertNegativePadsToSlice,
)
from furiosa.optimizer.frontend.onnx.transformer.convert_prelu_to_relu import ConvertPReluToRelu
from furiosa.optimizer.frontend.onnx.transformer.eliminate_redundant_shape_pattern import (
    EliminateRedundantShapePattern,
)
from furiosa.optimizer.frontend.onnx.transformer.fuse_batchnorm import FuseBatchNorm
from furiosa.optimizer.frontend.onnx.transformer.fuse_conv import FuseConv
from furiosa.optimizer.frontend.onnx.transformer.fuse_depth_to_space import FuseDepthToSpace
from furiosa.optimizer.frontend.onnx.transformer.fuse_gather_matmul import FuseGatherMatMul
from furiosa.optimizer.frontend.onnx.transformer.fuse_lp_normalization import FuseLpNormalization
from furiosa.optimizer.frontend.onnx.transformer.fuse_pad import FusePad
from furiosa.optimizer.frontend.onnx.transformer.fuse_redundant_reshape_pattern import (
    FuseRedundantReshapePattern,
)
from furiosa.optimizer.frontend.onnx.transformer.polish_model import PolishModel
from furiosa.optimizer.frontend.onnx.utils.version_checker import CheckVersion


def optimize_model(
    model: onnx.ModelProto,
    input_shapes: Optional[Dict[str, List[int]]] = None,
    opset_version: int = __OPSET_VERSION__,
) -> onnx.ModelProto:
    model = _transform([CheckVersion(opset_version).transform], model)
    model = _transform([PolishModel(input_shapes).transform], model)

    # TODO check if graph_transform should apply.
    model = _transform([_reify], model)
    return model


def _transform(
    transformers: List[Callable[[onnx.ModelProto], onnx.ModelProto]], model: onnx.ModelProto
) -> onnx.ModelProto:
    for transform in transformers:
        model = transform(model)
    return model


def _reify(model: onnx.ModelProto) -> onnx.ModelProto:
    transformers = [
        ConvertConv1dToConv2d().transform,
        FuseConv().transform,
        ConvertNegativePadsToSlice().transform,
        FusePad().transform,
        FuseBatchNorm().transform,
        FuseDepthToSpace().transform,
        FuseLpNormalization().transform,
        FuseRedundantReshapePattern().transform,
        FuseGatherMatMul().transform,
        EliminateRedundantShapePattern().transform,
        ConvertPReluToRelu().transform,
    ]
    return _transform(transformers, model)
