from __future__ import print_function

import argparse
from pathlib import Path
import tempfile

import onnx

from furiosa.common.utils import eprint, get_sdk_version
from furiosa.quantizer import __version__ as quantizer_version
from furiosa.quantizer.frontend.onnx import post_training_quantization_with_random_calibration
from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode
from furiosa.tools.compiler.api import compile

__version__ = get_sdk_version(__name__)


def validate(model_path: Path):
    """
    Validate a given model

    :param model_path: Model path
    :return: None
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        if not model_path.exists():
            eprint(f"ERROR: {model_path} does not exist")

        print(f"Using furiosa-quantizer {quantizer_version}")
        tmpfile = tempfile.NamedTemporaryFile()

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

        print("[Step 2] Checking the model can be compiled to a NPU program ...")
        try:
            compile(step1_output, step2_output, verbose=True, target_npu="warboy-2pe")
        except Exception as e:
            eprint("[Step 2] Failed\n")
            raise e
        print("[Step 2] Passed")


def main():
    parser = argparse.ArgumentParser(
        description="Checking if a model is compatible with furiosa-sdk"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to Model file (tflite, onnx, and other model formats are supported)",
    )
    args = parser.parse_args()
    validate(Path(args.model_path))


if __name__ == "__main__":
    main()
