from __future__ import print_function

import argparse
from pathlib import Path
import sys
import tempfile

import onnx

from furiosa.common.utils import eprint, get_sdk_version
from furiosa.quantizer import __version__ as quantizer_ver
from furiosa.quantizer.frontend.onnx import post_training_quantization_with_random_calibration
from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode
from furiosa.tools.compiler.api import compile

__version__ = get_sdk_version("furiosa.litmus")


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
        print("[Step 1] Checking if the model can be transformed into a quantized model ...")
        try:
            quantized_model = post_training_quantization_with_random_calibration(
                model=onnx.load_model(model_path),
                per_channel=True,
                static=True,
                mode=QuantizationMode.DFG,
                num_data=10,
            )
        except Exception as e:
            eprint("[Step 1] Failed\n")
            raise e
        print("[Step 1] Passed")

        step1_output = f"{tmpdir}/step1.onnx"
        step2_output = f"{tmpdir}/step2.enf"

        try:
            onnx.save_model(quantized_model, step1_output)
        except Exception as e:
            eprint("[ERROR] Fail to save the model\n")
            raise e

        print(
            f"[Step 2] Checking if the model can be compiled for the NPU family [{target_npu}] ..."
        )
        try:
            compile(step1_output, step2_output, verbose=verbose, target_npu=target_npu)
        except Exception as e:
            eprint("[Step 2] Failed\n")
            raise e
        print("[Step 2] Passed")


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
