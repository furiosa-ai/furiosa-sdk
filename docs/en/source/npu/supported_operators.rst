.. _SupportedOperators:

*************************************************
List of Supported Operators for NPU Acceleration
*************************************************

FuriosaAI NPU and SDK can accelerate the following operators, as supported by 
`Tensorflow Lite <https://www.tensorflow.org/lite>`_ model and `ONNX <https://onnx.ai/>`_. 

The names of the operators use `ONNX`_ as a reference.

.. note::

    If NPU acceleration is not supported, it will run on the CPU.
    For some operators, even if NPU acceleration is supported, if certain conditions are not met, they may be split into several operators 
    or may run on the CPU. Some examples of this would be when the weight of the model is larger than NPU memory, or if the NPU memory 
    is not sufficient to process a certain computation. 

.. list-table:: Operators Accelerated on NPU
   :widths: 50 200
   :header-rows: 1

   * - Name of operator 
     - Additional details 
   * - `Add <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add>`_
     -
   * - `AveragePool <https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool>`_
     -
   * - `BatchNormalization <https://github.com/onnx/onnx/blob/master/docs/Operators.md#batchnormalization>`_
     - Acceleration supported, only if after Conv 
   * - `Clip <https://github.com/onnx/onnx/blob/master/docs/Operators.md#clip>`_
     -
   * - `Concat <https://github.com/onnx/onnx/blob/master/docs/Operators.md#concat>`_
     - Acceleration supported, only for height axis
   * - `Conv <https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv>`_
     - Acceleration supported, only for `group` <= 128 and dilation <= 12
   * - `ConvTranspose <https://github.com/onnx/onnx/blob/master/docs/Operators.md#convtranspose>`_
     -
   * - `DepthToSpace <https://github.com/onnx/onnx/blob/master/docs/Operators.md#depthtospace>`_
     -
   * - `Exp <https://github.com/onnx/onnx/blob/master/docs/Operators.md#exp>`_
     -
   * - `Expand <https://github.com/onnx/onnx/blob/master/docs/Operators.md#expand>`_
     -
   * - `Flatten <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten>`_
     -
   * - `Gemm <https://github.com/onnx/onnx/blob/master/docs/Operators.md#gemm>`_
     -
   * - `LeakyRelu <https://github.com/onnx/onnx/blob/master/docs/Operators.md#leakyrelu>`_
     -
   * - `LpNormalization <https://github.com/onnx/onnx/blob/master/docs/Operators.md#lpnormalization>`_
     -  Acceleration supported, only for p = 2 and batch <= 2
   * - `MatMul <https://github.com/onnx/onnx/blob/master/docs/Operators.md#matmul>`_
     -
   * - `MaxPool <https://github.com/onnx/onnx/blob/master/docs/Operators.md#maxpool>`_
     -
   * - `Mean <https://github.com/onnx/onnx/blob/master/docs/Operators.md#mean>`_
     -
   * - `Mul <https://github.com/onnx/onnx/blob/master/docs/Operators.md#mul>`_
     -
   * - `Pad <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad>`_
     -
   * - `ReduceL2 <https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceL2>`_
     -
   * - `ReduceSum <https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum>`_
     -
   * - `Relu <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu>`_
     -
   * - `Reshape <https://github.com/onnx/onnx/blob/master/docs/Operators.md#reshape>`_
     -
   * - `Pow <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pow>`_
     -
   * - `SpaceToDepth <https://github.com/onnx/onnx/blob/main/docs/Operators.md#SpaceToDepth>`_
     - Acceleration supported, only for mode="CRD" and Furiosa SDK version 0.6.0 or higher

   * - `Sigmoid <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid>`_
     -
   * - `Slice <https://github.com/onnx/onnx/blob/master/docs/Operators.md#slice>`_
     - Acceleration supported, only for height axis
   * - `Softmax <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax>`_
     - Acceleration supported, only for batch <= 2
   * - `Softplus <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softplus>`_
     -
   * - `Sub <https://github.com/onnx/onnx/blob/master/docs/Operators.md#sub>`_
     -
   * - `Split <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Split>`_
     -
   * - `Sqrt <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt>`_
     -
   * - `Transpose <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose>`_
     -
   * - `Unsqueeze <https://github.com/onnx/onnx/blob/master/docs/Operators.md#unsqueeze>`_
     -
