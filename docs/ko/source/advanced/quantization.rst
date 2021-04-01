*************************************
모델 양자화
*************************************

FuriosaAI NPU인 Warboy는 8bit 정수형 모델만을 지원한다.
실수형 데이터 타입 기반의 모델을 지원하기 위해,
Furiosa SDK는 FP16, FP32 실수형 데이터 타입 기반 모델을 양자화(quantization)하여
8bit 정수형 데이터 타입 모델로 변환하는 도구를 제공한다.
양자화란 모델의 처리 성능을 높이거나 하드웨어를 가속하기 위해 사용되는 보편적인 기술로,
FuriosaAI SDK에서 제공하는 양자화 도구를 사용하면 NPU를 활용하여 더욱 다양한 모델을 가속할 수 있다.

FuriosaAI SDK가 지원하는 양자화 방식은  *post-training 8-bit quantization* 기반이며 
`Tensorflow Lite 8-bit quantization specification <https://www.tensorflow.org/lite/performance/quantization_spec>`_ 
을 따른다.

SDK가 제공하는 API와 명령 줄 도구를 사용하여 ONNX 모델을 8bit 양자화 모델로 변환할 수 있다.
사용 방법은 아래 링크에서 확인할 수 있다:

* `Python SDK Quantization 예제 <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/furiosa-sdk-quantizer>`_
* `Python 레퍼런스 - furiosa.quantizer <https://furiosa-ai.github.io/renegade-manual/references/python/quantizer/index.html>`_


TensorFlow 모델은 추후 지원할 예정이다.

동작 방식
======================================

양자화 도구는 아래 그림에서 표현된 바와 같이 ONNX 모델을 입력으로 받아
아래 3단계를 거쳐 양자화를 실행하고 양자화된 ONNX 모델을 출력한다.

#. Graph optimization 
#. Calibration
#. Quantization

.. image:: ../../../imgs/nux-quantizer_quantization_pipepline-edd29681.png
