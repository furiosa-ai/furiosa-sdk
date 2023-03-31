"""Furiosa Litmus, which readily checks whether a given model can be compiled with Furiosa SDK"""
import argparse
from pathlib import Path
import sys
import tempfile
from typing import Dict, Tuple

from google.protobuf.message import DecodeError
import numpy as np
import onnx

from furiosa.common.utils import eprint, get_sdk_version
from furiosa.optimizer import optimize_model
from furiosa.quantizer import CalibrationMethod, Calibrator
from furiosa.quantizer import __version__ as quantizer_ver
from furiosa.quantizer import quantize
from furiosa.tools.compiler.api import compile

__version__ = get_sdk_version("furiosa.litmus")


def calibrate_with_random_data(
    model: onnx.ModelProto, dataset_size: int = 8
) -> Dict[str, Tuple[float, float]]:
    """Estimates the range of tensors in a model, based on a random dataset.
    Args:
        model: An ONNX model to calibrate.
        dataset_size: the size of a random dataset to use.
    Returns:
        A dict mapping tensors in the model to their minimum and maximum values.
    """
    calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)
    initializers = set(tensor.name for tensor in model.graph.initializer)
    rng = np.random.default_rng()
    for _ in range(dataset_size):
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
                    raise RuntimeError(
                        f"The static shape of tensor '{value_info.name}' must be provided"
                    )
            np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[value_info.type.tensor_type.elem_type]
            if np.issubdtype(np_dtype, np.floating):
                inputs = rng.standard_normal(size=shape, dtype=np_dtype)
            elif np.issubdtype(np_dtype, np.integer):
                iinfo = np.iinfo(np_dtype)
                inputs = rng.integers(
                    iinfo.min, iinfo.max, size=shape, dtype=np_dtype, endpoint=True
                )
            else:
                elem_type = onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)
                raise NotImplementedError(
                    f"tensor '{value_info.name}' is of {elem_type} but a model whose input tensor is of {elem_type} cannot be randomly calibrated yet"
                )
        calibrator.collect_data([[inputs]])
    return calibrator.compute_range()


def validate(model_path: Path, verbose: bool, target_npu: str):
    """
    Validate a given model

    :param model_path: Model path
    :return: None
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        if not model_path.exists():
            eprint(f"ERROR: {model_path} does not exist")

        print(
            f"furiosa-quantizer {quantizer_ver.version} (rev. {quantizer_ver.hash[0:9]})",
            f"furiosa-litmus {__version__.version} (rev. {__version__.hash[0:9]})",
            file=sys.stderr,
        )
        # Try quantization on input models
        print("[Step 1] Checking if the model can be loaded and optimized ...", flush=True)
        try:
            onnx_model = onnx.load_model(model_path)
            onnx_model = optimize_model(onnx_model)
        except DecodeError as de:
            eprint("[Step 1] ERROR: The input file should be a valid ONNX file")
            raise SystemExit(de)
        except Exception as e:
            eprint("[Step 1] Failed\n")
            raise e
        print("[Step 1] Passed", flush=True)

        print("[Step 2] Checking if the model can be quantized ...", flush=True)
        try:
            ranges = calibrate_with_random_data(onnx_model)
            quantized_model = quantize(onnx_model, ranges)
        except Exception as e:
            eprint("[Step 2] Failed\n")
            raise e
        print("[Step 2] Passed", flush=True)

        print("[Step 3] Checking if the model can be saved as a file ...", flush=True)
        try:
            tmp_dfg_path = f"{tmpdir}/output.dfg"
            with open(tmp_dfg_path, "wb") as f:
                f.write(bytes(quantized_model))
        except Exception as e:
            eprint("[Step 3] Failed\n")
            raise e
        print("[Step 3] Passed", flush=True)

        print(
            f"[Step 4] Checking if the model can be compiled for the NPU family [{target_npu}] ...",
            flush=True,
        )
        try:
            compile(bytes(quantized_model), target_ir="enf", verbose=verbose, target_npu=target_npu)
        except Exception as e:
            eprint("[Step 4] Failed\n")
            raise e
        print("[Step 4] Passed")


def main():
    parser = argparse.ArgumentParser(
        description="Validate the model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to Model file (tflite, onnx, and other model formats are supported)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument(
        '--target-npu',
        type=str,
        default='warboy-2pe',
        help='Target NPU: warboy, warboy-2pe',
    )
    args = parser.parse_args()
    validate(Path(args.model_path), verbose=args.verbose, target_npu=args.target_npu)


if __name__ == "__main__":
    main()
