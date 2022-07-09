.. _ModelQuantization:

*************************************
Model Quantization
*************************************

Furiosa SDK and first generation Warboy support INT8 models.
To support floating point models, Furiosa SDK provides quantization tools to convert
FP16 or FP32 floating point data type models into INT8 data type models.
Quantization is a common technique used to increase model processing performance or accelerate hardware.
Using the quantization tool provied by Furiosa SDK, a greater variety of models can be accelerated by deploying the NPU.

Quantization method supported by Furiosa SDK is based on *post-training 8-bit quantization*, and follows
`Tensorflow Lite 8-bit quantization specification <https://www.tensorflow.org/lite/performance/quantization_spec>`_.

How It Works
======================================

As shown in the diagram below, the quantization tool receives the ONNX model as input,
performs quantization through the following three steps, and outputs the quantized ONNX model.

#. Graph Optimization
#. Calibration
#. Quantization

.. figure:: ../../../imgs/nux-quantizer_quantization_pipepline-edd29681.png
  :alt: Quantization Process
  :class: with-shadow
  :align: center

In the graph optimization process, the topological structure of the graph is changed by adding or replacing
operators in the model through analysis of the original model network structure,
so that the model can process quantized data with a minimal drop in accuracy.

In the calibration process, the data used to train the model is required in order to calibrate the weights of the model.


Accuracy of Quantized Models
========================================

Models - with their respective validation datasets - were quantized using min-max calibration with the Furiosa SDK quantizer.
The table below compares accuracies.

.. _QuantizationAccuracyTable:

.. list-table:: Quantization Accuracy
   :widths: 50 50 50 50
   :header-rows: 1

   * - Model
     - FP Accuracy (%)
     - INT8 Accuracy (%)
     - Accuracy Drop (%)
   * - ResNet50 v1.0
     - 76.456
     - 76.002
     - 0.594
   * - SSD MobileNet 300x300
     - 22.137
     - 22.815
     - 1.392
   * - SSD Resnet34 1200x1200
     - 22.308
     - 22.069
     - 1.071


Model Quantization APIs
========================================

You can use the APU and command line tool provided in this SDK to convert an ONNX model into an 8bit quantized model.

Refer to the links below for further instructions.

* `Python SDK example: How to use Furiosa SDK from start to finish <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/HowToUseFuriosaSDKFromStartToFinish.ipynb>`_
* `Python SDK Quantization example <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/quantizers>`_
* `Python reference - furiosa.quantizer <https://furiosa-ai.github.io/docs/v0.6.0/en/api/python/furiosa.quantizer.html>`_
