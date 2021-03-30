*************************************
모델 양자화
*************************************

FuriosaAI NPU인 Warboy는 8비트 정수 모델을 지원합니다.
그러나 Furiosa SDK는 기존 FP16, FP32 실수형 데이터타입 기반의 모델을 양자화(quantization) 
하여 8비트 정수형 데이터타입 모델로 변환하는 도구를 제공합니다.
양자화는 모델의 처리 성능을 높이거나 하드웨어 가속을 위해 사용되는 보편적인 기술로,
FuriosaAI SDK에서 제공하는 양자화 도구를 통해 더욱 다양한 모델을 NPU를 활용하여 가속할 수 있습니다.

FuriosaAI SDK가 지원하는 양자화 방식은  *post-training 8-bit quantization* 기반이며 
`Tensorflow Lite 8-bit quantization specification <https://www.tensorflow.org/lite/performance/quantization_spec>`_ 
을 따릅니다.

현재 Onnx 모델을 8비트 양자화 모델로 변환하는 API와 커맨드 라인 도구가 SDK를 통해 제공합니다. 
아래 링크에서 사용 방법을 참고하실 수 있습니다.

* `Python SDK Quantization 예제 <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/furiosa-sdk-quantizer>`_
* `Python 레퍼런스 - furiosa.quantizer <https://furiosa-ai.github.io/renegade-manual/references/python/quantizer/index.html>`_


TensorFlow 모델은 추후 지원할 예정입니다.

동작 방식
======================================

양자화 도구는 아래 그림에서 표현된 바와 같이 Onnx 모델을 입력으로 받아
아래 3단계를 거쳐 양자화를 실행하고 양자화된 Onnx 모델을 출력합니다.

#. Graph optimization 
#. Calibration
#. Quantization

.. image:: ../../../imgs/nux-quantizer_quantization_pipepline-edd29681.png
