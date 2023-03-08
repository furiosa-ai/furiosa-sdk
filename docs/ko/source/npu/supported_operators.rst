.. _SupportedOperators:

******************************************
NPU 가속 지원 연산자 목록
******************************************

FuriosaAI NPU와 SDK 에서는
`Tensorflow Lite <https://www.tensorflow.org/lite>`_ 모델과 `ONNX <https://onnx.ai/>`_ 가 지원하는
아래 연산자들을 가속할 수 있다. 연산자 이름은 `ONNX`_ 를 기준으로 한다.

.. note::

    NPU 가속을 지원하지 않는 경우에는 CPU에서 동작하게 된다.
    또한 NPU 가속을 지원하는 일부 연산자는 특정 조건을 만족하지 않을 경우 다수의 연산자로 분할되어 동작하거나
    CPU 에서 동작할 수 있다. 모델의 가중치가 NPU 메모리 보다 크거나 NPU의 메모리로
    특정 연산을 처리하기에 부족한 경우가 한 가지 예이다.

.. list-table:: NPU 가속 지원 연산자
   :widths: 50 200
   :header-rows: 1

   * - 연산자 이름
     - 추가 설명
   * - `Add <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add>`_
     -
   * - `AveragePool <https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool>`_
     -
   * - `BatchNormalization <https://github.com/onnx/onnx/blob/master/docs/Operators.md#batchnormalization>`_
     - Conv 다음에 있는 경우에 한하여 가속 지원
   * - `Clip <https://github.com/onnx/onnx/blob/master/docs/Operators.md#clip>`_
     -
   * - `Concat <https://github.com/onnx/onnx/blob/master/docs/Operators.md#concat>`_
     - H axis 지원 (>= 0.6.0), C axis 지원 (>= 0.7.0)
   * - `Conv <https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv>`_
     - `group` <= 128, `dilation` <= 12 인 경우에 대해 지원
   * - `ConvTranspose <https://github.com/onnx/onnx/blob/master/docs/Operators.md#convtranspose>`_
     -
   * - `DepthToSpace <https://github.com/onnx/onnx/blob/master/docs/Operators.md#depthtospace>`_
     - CRD 모드 지원 (>= 0.6.0), DCR 모드 지원 (>= 0.7.0)
   * - `Exp <https://github.com/onnx/onnx/blob/master/docs/Operators.md#exp>`_
     - 0.7.0 이상 부터 지원
   * - `Elu <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Elu>`_
     - 0.7.0 이상 부터 지원
   * - `Erf <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Erf>`_
     - 0.7.0 이상 부터 지원
   * - `Expand <https://github.com/onnx/onnx/blob/master/docs/Operators.md#expand>`_
     -
   * - `Flatten <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten>`_
     -
   * - `Gemm <https://github.com/onnx/onnx/blob/master/docs/Operators.md#gemm>`_
     -
   * - `Gelu <https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.Gelu>`_
     - 0.7.0 이상 부터 지원
   * - `LeakyRelu <https://github.com/onnx/onnx/blob/master/docs/Operators.md#leakyrelu>`_
     -
   * - `Log <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log>`_
     -
   * - `LpNormalization <https://github.com/onnx/onnx/blob/master/docs/Operators.md#lpnormalization>`_
     - p = 2 인 경우에 한하여 지원
   * - `MatMul <https://github.com/onnx/onnx/blob/master/docs/Operators.md#matmul>`_
     -
   * - `MaxPool <https://github.com/onnx/onnx/blob/master/docs/Operators.md#maxpool>`_
     -
   * - `Mean <https://github.com/onnx/onnx/blob/master/docs/Operators.md#mean>`_
     -
   * - `Mul <https://github.com/onnx/onnx/blob/master/docs/Operators.md#mul>`_
     -
   * - `Pad <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad>`_
     - CWH axis 지원 (>= 0.7.0)
   * - `ReduceL2 <https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceL2>`_
     -
   * - `ReduceSum <https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum>`_
     -
   * - `Relu <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu>`_
     -
   * - `Reshape <https://github.com/onnx/onnx/blob/master/docs/Operators.md#reshape>`_
     -
   * - `Resize <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize>`_
     - Linear, Nearest 모드 지원 (>= 0.7.0)
   * - `Pow <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pow>`_
     -
   * - `SpaceToDepth <https://github.com/onnx/onnx/blob/main/docs/Operators.md#SpaceToDepth>`_
     - CRD 모드 지원 (>= 0.6.0), DCR 모드 지원 (>= 0.7.0)
   * - `Sigmoid <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid>`_
     -
   * - `Slice <https://github.com/onnx/onnx/blob/master/docs/Operators.md#slice>`_
     - H axis 지원 (>= 0.6.0), C axis 지원 (>= 0.7.0)
   * - `Softmax <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax>`_
     -
   * - `Softplus <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softplus>`_
     -
   * - `Sub <https://github.com/onnx/onnx/blob/master/docs/Operators.md#sub>`_
     -
   * - `Split <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Split>`_
     -
   * - `Sqrt <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt>`_
     -
   * - `Tanh <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh>`_
     - 0.7.0 이상 버전 부터 지원
   * - `Transpose <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose>`_
     - 0.6.0 이상 버전 부터 지원
   * - `Unsqueeze <https://github.com/onnx/onnx/blob/master/docs/Operators.md#unsqueeze>`_
     -
