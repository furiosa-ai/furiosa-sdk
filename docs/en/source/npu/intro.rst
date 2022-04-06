**********************************
FuriosaAI NPU
**********************************

FuriosaAI NPU is a chip with an architecture optimized for deep learning inference.
It demonstrates high deep learning inference performance while remaining cost-efficient.
FuriosaAI NPU is optimized for inferences with low batch sizes; for inference requests with low batch sizes,
all of the chip's resources are maximally utilized to achieve low latency.

The large on-chip memory is also able to retain most major CNN models, thereby eliminating memory bottlenecks,
and achieving high energy efficiency.

FuriosaAI NPU supports key CNN models used in various vision tasks, including
Image Classification, Object Detection, OCR, and Super Resolution.
In particular, the chip demonstrates superior performance in computations such as depthwise/group convolution,
that drive high accuracy and computational efficiency in state-of-the-art CNN models.


.. _IntroToWarboy:

**********************************
FuriosaAI Warboy
**********************************

FuriosaAI's first generation NPU Warboy, delivers 64 TOPS performance and includes 32MB of SRAM.
Warboy consists of two processing elements (PE), which each delivers 32 TOPS performance and can be deployed independently.
With a total performance of 64 TOPS, should there be a need to maximize response speed to models, the two PEs may undergo fusion,
to aggregate as a larger, single PE. Depending on the users' model size or performance requirements the PEs may be 1) fused
so as to optimize response time, or 2) utilized independently to optimize for throughput.

FuriosaAI SDK provides the compiler, runtime software, and profiling tools for the FuriosaAI NPU.
It also supports the INT8 quantization scheme, used as a standard in TensorFLow and PyTorch, while providing tools to convert Floating Point models using Post Training Quantization.
With the FuriosaAI SDK, users can compile trained or exported models in formats commonly used for inference (TensorFlowLite or ONNX), and accelerate them on FuriosaAI NPU.

FuriosaAI Warboy HW Specifications
----------------------------------
The chip is built with 5 billion transistors, dimensions of 180mm^2, clock speed of 2GHz, and delivers peak performance of 64 TOPS of INT8.
It also supports a maximum of 4266 for LPDDR4x. Warboy has a DRAM bandwidth of 66GB/s, and supports PCIe Gen4 8x.

.. figure:: ../../../imgs/warboy_spec.png
  :alt: Warboy Hardware Specification
  :class: with-shadow
  :align: center
  :width: 500

  Warboy (rev. a0) Specification

FuriosaAI Warboy Performance
------------------------------
Results submitted to MLCommons can be found at
`MLPerf™ Inference Edge v1.1 Results <https://mlcommons.org/en/inference-edge-11/>`_

See Also
=================================
* `MLPerf™ Inference Edge v1.1 Results <https://mlcommons.org/en/inference-edge-11/>`_