#!/usr/bin/env python3

"""Image classification example"""
import sys
import onnx
from pathlib import Path

from furiosa_sdk_quantizer.frontend import onnx as quantizer
from furiosa_sdk_quantizer.frontend.onnx.quantizer.utils import QuantizationMode


def quantize(fp32_model_path, output_model_path, num_calib):
    model = onnx.load_model(fp32_model_path)
    print(model.opset_import)
    quantized_model = quantizer.post_training_quantization_with_random_calibration(model,
                                                                                   True,
                                                                                   True,
                                                                                   QuantizationMode.dfg,
                                                                                   num_data=num_calib)
    onnx.save_model(quantized_model, output_model_path)



if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("./quantize.py <fp32_model_path> <output_model_path> <num_random_data>\n")
        sys.exit(-1)

    fp32_model_path = Path(sys.argv[1])
    if not fp32_model_path.exists():
        sys.stderr.write(f"{fp32_model_path} not found")
        sys.exit(-1)
    output_model_path = Path(sys.argv[2])

    if len(sys.argv) >= 4:
        num_calib = int(sys.argv[3])
    else:
        num_calib = 5

    quantize(fp32_model_path, output_model_path, num_calib=num_calib)
