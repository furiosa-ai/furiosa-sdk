#!/usr/bin/env python3

from pathlib import Path
import unittest

import numpy as np
import onnx

from furiosa_sdk_quantizer.frontend.onnx.quantizer.calibrator import ONNXCalibrator


class ONNXCalibratorTest(unittest.TestCase):
    def test_calibrate(self):
        model = onnx.load(Path(__file__).resolve().parent / "test_calibrate.onnx")
        calibrator = ONNXCalibrator(model)
        rng = np.random.default_rng()
        calibration_dataset = [
            {"input": rng.random((1, 3, 224, 224), np.float32)},
            {"input": rng.random((1, 3, 224, 224), np.float32)},
            {"input": rng.random((1, 3, 224, 224), np.float32)},
        ]

        ranges = calibrator.calibrate(calibration_dataset)
        self.assertEqual(7, len(ranges))


if __name__ == "__main__":
    unittest.main()
