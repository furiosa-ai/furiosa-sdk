import os
import tempfile
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import onnx
import onnxruntime.quantization.calibrate
import tqdm


class CalibrationError(Exception):
    """The base class for all exceptions that are related to calibration."""


class CalibrationDataReaderForIterator(onnxruntime.quantization.calibrate.CalibrationDataReader):
    """A CalibrationDataReader that wraps dicts mapping input tensor names to their values."""

    def __init__(self, iterator: Iterator[Dict[str, np.ndarray]]):
        self.iterator = iterator

    def get_next(self):
        return next(self.iterator, None)


def calibrate(
    model: onnx.ModelProto,
    dataset: Iterable[Dict[str, np.ndarray]],
    augmented_model_path: Optional[str] = None,
) -> Dict[str, Tuple[float, float]]:
    """Estimates the range of tensors in a model, based on a dataset.

    Args:
        model: An ONNX model to calibrate.
        dataset: An Iterable that returns dicts mapping input tensor names to their values.
        augmented_model_path: A path to save an augmented model to.

    Returns:
        A dict mapping tensors in the model to their minimum and maximum values.
    """
    if augmented_model_path is None:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            augmented_model_path = f.name

    calibrator = onnxruntime.quantization.calibrate.MinMaxCalibrater(
        model, augmented_model_path=augmented_model_path
    )
    if os.environ.get("TQDM_DISABLE"):
        dataset = iter(dataset)
    else:
        dataset = tqdm.tqdm(dataset, desc="Calibration")
    calibrator.collect_data(CalibrationDataReaderForIterator(iter(dataset)))
    return calibrator.compute_range()


def calibrate_with_random_data(
    model: onnx.ModelProto, dataset_size: int = 8, augmented_model_path: Optional[str] = None
) -> Dict[str, Tuple[float, float]]:
    """Estimates the range of tensors in a model, based on a random dataset.

    Args:
        model: An ONNX model to calibrate.
        dataset_size: the size of a random dataset to use.
        augmented_model_path: A path to save an augmented model to.

    Returns:
        A dict mapping tensors in the model to their minimum and maximum values.
    """
    if augmented_model_path is None:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            augmented_model_path = f.name

    calibrator = onnxruntime.quantization.calibrate.MinMaxCalibrater(
        model, augmented_model_path=augmented_model_path
    )
    initializers = set(tensor.name for tensor in model.graph.initializer)
    rng = np.random.default_rng()
    for _ in range(dataset_size):
        inputs = {}
        for value_info in model.graph.input:
            if value_info.name in initializers:
                continue
            # https://github.com/onnx/onnx/blob/master/docs/IR.md#static-tensor-shapes
            #
            # > The static shape is defined by 'TensorShapeProto':
            # >
            # >     message TensorShapeProto {
            # >       message Dimension {
            # >         oneof value {
            # >           int64 dim_value = 1;
            # >           string dim_param = 2;
            # >         };
            # >       };
            # >       repeated Dimension dim = 1;
            # >     }
            # >
            # > Which is referenced by the Tensor type message:
            # >
            # >     message Tensor {
            # >       optional TensorProto.DataType elem_type = 1;
            # >       optional TensorShapeProto shape = 2;
            # >     }
            # >
            # > The empty list of dimension sizes, [], is a valid tensor shape, denoting a
            # > zero-dimension (scalar) value. A zero-dimension tensor is distinct from a tensor of
            # > unknown dimensionality, which is indicated by an absent 'shape' property in the
            # > Tensor message. When the shape property is absent in the type of a value (including
            # > node input), it indicates that the corresponding runtime value may have any shape.
            # > This sub-section describes how to interpret a missing-shape or a shape with missing
            # > dimensions etc. However, specific usage contexts may impose further constraints on a
            # > type and shape. For example, the inputs and outputs of a model (top-level graph) are
            # > required to have a shape, indicating the rank of inputs and outputs, even though the
            # > exact dimensions need not be specified.
            shape = []
            for dimension in value_info.type.tensor_type.shape.dim:
                if dimension.HasField("dim_value"):
                    shape.append(dimension.dim_value)
                else:
                    raise CalibrationError(
                        f"The static shape of tensor '{value_info.name}' must be provided"
                    )
            if shape:
                if value_info.type.tensor_type.elem_type == onnx.TensorProto.DataType.FLOAT:
                    inputs[value_info.name] = rng.standard_normal(size=shape, dtype=np.float32)
                else:
                    raise NotImplementedError(
                        onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)
                    )
            else:
                if value_info.type.tensor_type.elem_type == onnx.TensorProto.DataType.FLOAT:
                    inputs[value_info.name] = rng.standard_normal(dtype=np.float32)
                else:
                    raise NotImplementedError(
                        onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)
                    )
        calibrator.collect_data(CalibrationDataReaderForIterator(iter([inputs])))
    return calibrator.compute_range()
