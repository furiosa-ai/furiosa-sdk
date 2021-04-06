******************************************
NPU 가속 연산자 목록
******************************************

FuriosaAI NPU와 SDK 0.2.1 에서는 
`Tensorflow Lite <https://www.tensorflow.org/lite>`_ 모델과 `ONNX <https://onnx.ai/>`_ 모델에 
포함된 아래 28개 연산자를 가속할 수 있다. 아래 연산자 이름은 `ONNX`_ 를 기준으로 한다.

* Add
* AveragePool
* Clip
* Concat
* Conv
* ConvTranspose
* DepthToSpace
* Exp
* Expand
* Flatten
* Gemm
* LpNormalization (when p = 2)
* MatMul
* MaxPool
* Mean
* Mul
* Pad
* ReduceL2
* ReduceSum
* Relu
* Reshape
* Sigmoid
* Slice
* Softmax
* Softplus
* Split
* Transpose
* Unsqueeze