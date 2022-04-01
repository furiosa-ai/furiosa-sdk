**********************************
FuriosaAI NPU
**********************************

The FuriosaAI NPU is dedicated to DNN inference tasks. Warboy employs a domain-specific design and a custom compiler
specialized for inference processing of deep neural network models. Even on relatively small batch sizes, FuriosaAI
NPU achieves a small latency of inference tasks by high utilization of HW resources. With retaining DNN models on the
on-chip SRAM, memory I/O bottleneck can be removed and high energy efficiency can be achieved. Especially
Depthwise/Group Convolution can be remarkably accelerated on FuriosaAI NPU, which enables high accuracy and
computational efficiency of state-of-the-art CNN models. The first generation FuriosaAI NPU, Warboy accelerates
vision intelligence including super resolution, optical character recognition, traffic management, intelligent video
analytics, industrial safety, metaverse, and autonomous driving.

.. _IntroToWarboy:

**********************************
FuriosaAI Warboy
**********************************

The first generation FuriosaAI NPU, Warboy shows 64 TOPS (INT8) peak performance and has 32MB on-chip SRAM.
Warboy consists of two processing elements (PE). Each PE can be used independent AI chip of 32 TOPS for higher throughput.
The two PEs can be fused as a whole AI chip of 64 TOPS.
The PE fusion can be determined according to either the given model size or latency requirements.

FuriosaAI SDK provides NPU compiler, runtime, and profiling tools.
Furiosa quantizer supports the INT8 Quantization Scheme used commonly in TensorFlow and PyTorch.
Floating point models can be quantized by Furiosa quantizer with post training quantization (PTQ).
FuriosaAI SDK makes TensorFlowLite and ONNX format models be accelerated on FuriosaAI NPU.

FuriosaAI Warboy Hardware Specification
----------------------------------

.. figure:: ../../../imgs/warboy_spec.png
  :alt: Warboy Hardware Specification
  :class: with-shadow
  :align: center

FuriosaAI Warboy Performance
------------------------------
The FuriosaAI Warboy MLPerf performance results can be shown on the below:
`MLPerf™ Inference Edge v1.1 Results <https://mlcommons.org/en/inference-edge-11/>`_

References
=================================
* `MLPerf™ Inference Edge v1.1 Results <https://mlcommons.org/en/inference-edge-11/>`_