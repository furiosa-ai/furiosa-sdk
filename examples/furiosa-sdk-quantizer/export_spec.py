#!/usr/bin/env python3

"""Export the spec"""
import sys
from pathlib import Path

import onnx
from furiosa.quantizer.frontend.onnx import export_spec


def do_export_spec(fp32_model_path, output_path):
    model = onnx.load_model(fp32_model_path)
    print(model.opset_import)
    with open(output_path, "w") as output:
        export_spec(model, output)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("./export_spec.py <fp32_model_path> <spec_output_path>\n")
        sys.exit(-1)

    fp32_model_path = Path(sys.argv[1])
    if not fp32_model_path.exists():
        sys.stderr.write(f"{fp32_model_path} not found")
        sys.exit(-1)
    spec_output_path = Path(sys.argv[2])

    do_export_spec(fp32_model_path, spec_output_path)
