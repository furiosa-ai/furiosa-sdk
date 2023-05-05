.. _ModelQuantization:

*************************************
모델 양자화
*************************************

FuriosaAI SDK와 1세대 Warboy는 8bit 정수형 모델을 지원한다.
실수형 데이터 타입 기반의 모델을 지원하기 위해 Furiosa SDK는 FP16, FP32 실수형 데이터 타입 기반 모델을 양자화(quantization)하여
8bit 정수형 데이터 타입 모델로 변환하는 도구를 제공한다.
양자화란 모델의 처리 성능을 높이거나 하드웨어를 가속하기 위해 사용되는 보편적인 기술로
FuriosaAI SDK에서 제공하는 양자화 도구를 사용하면 NPU를 활용하여 더욱 다양한 모델을 가속할 수 있다.

FuriosaAI SDK가 지원하는 양자화 방식은  *post-training 8-bit quantization* 기반이며
`Tensorflow Lite 8-bit quantization specification <https://www.tensorflow.org/lite/performance/quantization_spec>`_
을 따른다.

동작 방식
======================================

양자화 도구는 아래 그림에서 표현된 바와 같이 ONNX 모델을 입력으로 받아
아래 3단계를 거쳐 양자화를 실행하고 양자화된 ONNX 모델을 출력한다.

#. 그래프 최적화(Graph Optimization)
#. 보정(Calibration)
#. 양자화(Quantization)

.. figure:: ../../../imgs/nux-quantizer_quantization_pipepline-edd29681.png
  :alt: Quantization Process
  :class: with-shadow
  :align: center

그래프 최적화 과정에서는 모델이 양자화된 데이터를 정확도 저하를 최소화하면서 처리할 수 있도록
원본 모델 네트워크의 구조를 분석하여 모델에 연산자를 추가하거나 대체하여 그래프의 위상구조를 변경한다.

보정 과정에서는 데이터를 기반으로 모델의 가중치를 보정하며 이 과정에서
모델을 학습할 때 사용했던 데이터가 필요하다.


양자화 모델의 정확도
========================================

아래 표는 FuriosaAI SDK에서 제공하는 Quantizer와 다양한 보정 방법을 이용해 여러 모델을 양자화하고 원본 소수점 모델과 정확도를 비교한 것이다.

.. _QuantizationAccuracyTable:

.. list-table:: Quantization Accuracy
   :header-rows: 1

   * - Model
     - FP Accuracy
     - INT8 Accuracy (Calibration Method)
     - INT8 Accuracy ÷ FP Accuracy
   * - ConvNext-B
     - 85.8%
     - 80.376% (Asymmetric MSE)
     - 93.678%
   * - EfficientNet-B0
     - 77.698%
     - 73.556% (Asymmetric 99.99%-Percentile)
     - 94.669%
   * - EfficientNetV2-S
     - 84.228%
     - 83.566% (Asymmetric 99.99%-Percentile)
     - 99.214%
   * - ResNet50 v1.5
     - 76.456%
     - 76.228% (Asymmetric MSE)
     - 99.702%
   * - RetinaNet
     - mAP 0.3757
     - mAP 0.37373 (Symmetric Entropy)
     - 99.476%
   * - SSD MobileNet
     - mAP 0.23
     - mAP 0.23215 (Symmetric Min-Max)
     - 100.93%
   * - SSD ResNet34
     - mAP 0.20
     - mAP 0.21626 (Asymmetric Min-Max)
     - 108.13%
   * - YOLOX-l
     - mAP 0.497
     - mAP 0.48524 (Asymmetric 99.99%-Percentile)
     - 97.634%
   * - YOLOv5-l
     - mAP 0.490
     - mAP 0.47443 (Asymmetric MSE)
     - 96.822%
   * - YOLOv5-m
     - mAP 0.454
     - mAP 0.43963 (Asymmetric SQNR)
     - 96.835%


모델 양자화 APIs
========================================

SDK가 제공하는 API와 명령행 도구를 사용하여 ONNX 모델을 8bit 양자화 모델로 변환할 수 있다.
사용 방법은 아래에서 찾아볼 수 있다.

* `Python SDK 예제: 모델 생성 부터 인퍼런스 까지 <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/HowToUseFuriosaSDKFromStartToFinish.ipynb>`_
* `Python SDK Quantization 예제 <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/quantizers>`_
* `Python 레퍼런스 - furiosa.quantizer <https://furiosa-ai.github.io/docs/latest/en/api/python/furiosa.quantizer.html>`_
