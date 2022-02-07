#!/usr/bin/env python3

from pathlib import Path
import unittest

import numpy as np
import onnx
import onnx.numpy_helper

import furiosa.quantizer.frontend.onnx.calibrate


class CalibrateTest(unittest.TestCase):
    def test_calibrate_1(self):
        current_dir = Path(__file__).resolve().parent
        model = onnx.load(current_dir / "mnist/model.onnx")
        dataset = []
        for tensor_file in current_dir.glob("mnist/test_data_set_*/input_0.pb"):
            with open(tensor_file, "rb") as f:
                tensor = onnx.TensorProto()
                tensor.ParseFromString(f.read())
                # https://github.com/onnx/models/tree/master/vision/classification/mnist#preprocessing
                #
                # > # Preprocessing
                # >
                # > Images are resized into (28x28) in grayscale, with a black background and a
                # > white foreground (the number should be in white). Color value is scaled to
                # > [0.0, 1.0].
                dataset.append({"Input3": onnx.numpy_helper.to_array(tensor) / 255})
        self.assertAlmostEqual(
            furiosa.quantizer.frontend.onnx.calibrate.calibrate(model, dataset),
            {
                'Input3': (0.0, 1.0),
                'Convolution28_Output_0': (-5.180102348327637, 3.379333019256592),
                'Plus30_Output_0': (-5.088460922241211, 3.143223762512207),
                'ReLU32_Output_0': (0.0, 3.143223762512207),
                'Pooling66_Output_0': (0.0, 3.143223762512207),
                'Convolution110_Output_0': (-15.250411987304688, 7.669680595397949),
                'Plus112_Output_0': (-15.391451835632324, 7.254939556121826),
                'ReLU114_Output_0': (0.0, 7.254939556121826),
                'Pooling160_Output_0': (0.0, 7.254939556121826),
                'Pooling160_Output_0_reshape0': (0.0, 7.254939556121826),
                'Parameter193_reshape1': (-0.759514331817627, 1.1861310005187988),
                'Times212_Output_0': (-12.454174041748047, 25.349567413330078),
                'Plus214_Output_0': (-12.446382522583008, 25.417667388916016),
            },
        )

    def test_calibrate_2(self):
        current_dir = Path(__file__).resolve().parent
        model = onnx.load(current_dir / "mnist/model.onnx")
        rng = np.random.default_rng()
        # https://github.com/onnx/models/tree/master/vision/classification/mnist#inference
        #
        # > # Input
        # >
        # > Input tensor has shape (1x1x28x28), with type of float32. One image at a time. This
        # > model doesn't support mini-batch.
        # >
        # > # Preprocessing
        # >
        # > Images are resized into (28x28) in grayscale, with a black background and a white
        # > foreground (the number should be in white). Color value is scaled to [0.0, 1.0].
        fake_mnist = rng.integers(low=0, high=255, size=(8, 28, 28), dtype=np.uint8)
        dataset = [
            {"Input3": np.asarray(image, dtype=np.float32)[np.newaxis, np.newaxis, ...] / 255}
            for image in fake_mnist
        ]
        self.assertEqual(
            len(furiosa.quantizer.frontend.onnx.calibrate.calibrate(model, dataset)), 13
        )

    def test_calibrate_with_random_data(self):
        current_dir = Path(__file__).resolve().parent
        model = onnx.load(current_dir / "mnist/model.onnx")
        self.assertEqual(
            len(furiosa.quantizer.frontend.onnx.calibrate.calibrate_with_random_data(model)), 13
        )


if __name__ == "__main__":
    unittest.main()
