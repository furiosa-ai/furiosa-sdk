from __future__ import print_function

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

import onnx
from furiosa_sdk_quantizer.frontend.onnx import post_training_quantization_with_random_calibration
from furiosa_sdk_quantizer.frontend.onnx.quantizer.utils import QuantizationMode
from furiosa_sdk_runtime import session

utils = importlib.import_module('furiosa').utils
__version__ = utils.get_sdk_version(__name__)


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def validate(model_path: Path):
    """
    Validate a given model

    :param model_path: Model path
    :return: None
    """
    tmpfile = tempfile.NamedTemporaryFile()

    if not model_path.exists():
        _eprint(f'ERROR: {model_path} does not exist')

    # Try quantization on input models
    print(f'[Step 1] Checking if the model can be transformed into a quantized model ...')
    try:
        quantized_model = post_training_quantization_with_random_calibration(model=onnx.load_model(model_path),
                                                                             per_channel=True,
                                                                             static=True,
                                                                             mode=QuantizationMode.dfg,
                                                                             num_data=10)
    except Exception as e:
        _eprint("[Step 1] Failed\n")
        raise e
    print(f'[Step 1] Passed)')

    try:
        onnx.save_model(quantized_model, tmpfile.name)
    except Exception as e:
        _eprint("[ERROR] Fail to save the model\n")
        raise e

    print(f'[Step 2] Checking the model can be compiled to a NPU program ...')
    try:
        with open(tmpfile.name, "rb") as model_file:
            model = model_file.read()
            session.create(model=model)
    except Exception as e:
        _eprint("[Step 2] Failed\n")
        raise e
    print(f'[Step 2] Passed)')


def main():
    parser = argparse.ArgumentParser(description='Validate the model')
    parser.add_argument('model_path', type=str,
                             help='Path to Model file (tflite, onnx, and other model formats are supported)')
    args = parser.parse_args()
    validate(Path(args.model_path))


if __name__ == '__main__':
    main()
