#!/usr/bin/env python3

from pathlib import Path
import unittest

import onnx
import pickle

from furiosa_sdk_quantizer.frontend.onnx import post_training_quantize


class ONNXTest(unittest.TestCase):
    def test_post_training_quantize(self):
        model = onnx.load(
            Path(__file__).resolve().parent / "efficientdet_d0-f3276ba8.onnx"
        )
        # val2017-10.pickle contains a calibration dataset that consists
        # of 10 images in the COCO validation dataset [val2017.zip][].
        #
        # [val2017.zip]: http://images.cocodataset.org/zips/val2017.zip
        with open(Path(__file__).resolve().parent / "val2017-10.pickle", "rb") as f:
            dataset = pickle.load(f)
        model = post_training_quantize(model, dataset)


if __name__ == "__main__":
    unittest.main()
