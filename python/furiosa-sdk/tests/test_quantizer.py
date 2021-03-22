import unittest
import os

import numpy as np
import onnx

import se


class Dss(unittest.TestCase):
    @staticmethod
    def load_test_calibration_model():
        return onnx.load_model(os.path.join('examples', 'models', 'test_calibration_model.onnx'))

    def test_calibrate_multi_batch_type(self):
        calibration_model = Dss.load_test_calibration_model()
        # 35 elements of calibration data
        calibration_data = {'input': np.random.random((35, 3, 224, 224)).astype(np.float32)}
        dynamic_ranges = nux.quantizer.calibrate(calibration_model, calibration_data)
        self.assertEqual(7, len(dynamic_ranges))

    def test_calibrate_single_batch_type(self):
        calibration_model = Dss.load_test_calibration_model()
        # 35 elements of calibration data
        calibration_data = []
        for _ in range(35):
            calibration_data.append({'input': np.random.random((1, 3, 224, 224)).astype(np.float32)})
        dynamic_ranges = nux.quantizer.calibrate(calibration_model, calibration_data)
        self.assertEqual(7, len(dynamic_ranges))


if __name__ == '__main__':
    unittest.main()
