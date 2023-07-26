.. _ModelQuantization:

*************************************
모델 양자화
*************************************

모델 양자화는 모델 추론에 사용되는 컴퓨팅 리소스를 줄여 처리 성능을 높이거나 하드웨어를 가속하기 위해 사용되는 보편적인 기술이다.
모델에 사용되는 weight와 activation 텐서를 높은 정밀도 대신 낮은 정밀도로 대체하는 방식이다.

FuriosaAI SDK와 1세대 NPU인 Warboy는 8bit 정수형 모델만을 지원한다.
실수형 데이터 타입 기반의 모델을 지원하기 위해 FuriosaAI SDK는 FP16, FP32 실수형 데이터 타입 기반 모델을 양자화(quantization) 하여
8bit 정수형 데이터 타입 모델로 변환하는 도구를 제공한다. 이 도구를 사용하면 NPU를 활용하여 다양한 모델을 가속할 수 있다.

FuriosaAI SDK가 지원하는 양자화 방식은  *post-training 8-bit quantization* 기반이며
`Tensorflow Lite 8-bit quantization specification <https://www.tensorflow.org/lite/performance/quantization_spec>`_
을 따른다.


텐서의 데이터 타입
======================================

일반적으로 AI 모델은 FP32 타입을 기반으로 정의되고 있다. FP32 타입은 표현할 수 있는 범위가 넓고 수를 정밀하게 표현할 수 있다.
반면 Int8 타입은 표현할 수 있는 범위가 좁고 표현 가짓수가 256개에 불과하다.
그러나 추론 단계의 관점에서는 몇 가지 이유로 FP32보다 Int8이 유리한 점이 있다.

* 실수형 타입의 연산보다 정수형 타입의 연산이 빠르다.
* 실수형 타입의 연산보다 정수형 타입의 연산의 컴퓨팅 파워가 적게 든다.
* 실수형 타입의 값보다 정수형 타입의 값이 메모리를 적게 사용한다.

.. note::

    그 밖의 타입 (Warboy 미지원)

    * FP16: 실수형 타입이며 부호 1비트, 지수 5비트, 가수 10비트를 사용
    * BF16: 실수형 타입이며 부호 1비트, 지수 8비트, 가수 7비트를 사용
    * Int4: 정수형 타입이며 수를 표현하는데 4비트를 사용


양자화에 대한 산술 표현
======================================

* 실수형 타입과 정수형 타입 간의 산술적 관계는 다음 수식을 통해 표현할 수 있다. 정숫값에 zeropoint를 빼고 scale을 곱하면 실숫값을 계산할 수 있다.

.. figure:: ../../../imgs/quantization/equation_0.png
  :class: with-shadow
  :align: center


* 정수 Xq가 Int8 타입이면 표현할 수 있는 실수 범위는 다음과 같다.

.. figure:: ../../../imgs/quantization/equation_1.png
  :class: with-shadow
  :align: center


* 실수형 타입에서 정수형 타입으로의 변환은 다음 수식을 통해 표현할 수 있다.

.. figure:: ../../../imgs/quantization/equation_2.png
  :class: with-shadow
  :align: center

한계점
--------------------------------------

* 실수형 타입의 값을 정수형에 대응시키면 표현할 수 없는 값들이 존재하게 된다.
* 예를 들어, scale은 1.5이고 zeropoint는 0이라고 가정하자.

    * 실수 1.5에 대한 양자화된 정수는 1이고 실수 3.0에 대한 양자화된 정수는 2가 된다.
    * 이때 1.5와 3.0 사이에 있는 값들은 양자화 시 1.5나 3.0에 해당하는 1 또는 2가 될 수밖에 없으므로 오차가 발생하게 된다.
    * 이를 양자화 오류라고 부른다.

* Int8 양자화의 경우 일반적으로 다음 범위의 오차가 발생한다.

.. figure:: ../../../imgs/quantization/equation_3.png
  :class: with-shadow
  :align: center

양자화 schemes
======================================

대칭 양자화(Symmetric Quantization)
--------------------------------------

* 표현할 수 있는 실수의 범위가 0을 기준으로 대칭인 형태이다.
* 실수 범위로 [-a, a]를 표현하고 Int8 범위에서는 [-127, 127]로 사용한다.
* zeropoint를 0으로 고정할 수 있어서, Z에 대한 추가 연산을 줄일 수 있다.
* 이 경우 0을 기준으로 대칭을 유지하게 되므로 양수나 음수 중 절댓값이 큰 값이 존재한다면 반대 부호 쪽에서도 그 범위를 포함해야 된다.

    * 범위가 불필요하게 넓어짐에 따라 실수에 대한 표현력이 낮아지고 scale 값이 커져 양자화 오차도 커지게 된다.
    * 예를 들어 Relu 연산의 결과로 올 수 있는 실수 범위는 [0, a]이다. 이때 대칭 양자화를 적용하면 범위가 [-a, a]가 되기 때문에 불필요한 음수 영역의 확장으로 인해 실수 오차가 커지게 된다.


비대칭 양자화(Asymmetric Quantization)
--------------------------------------

* 임의의 실수를 표현할 수 있는 범위를 갖고 있다.
* 보다 정교하게 양자화할 수 있도록 실수 범위를 정할 수 있다.
* Warboy는 비대칭 양자화를 기본으로 지원한다.


양자화 매개변수
======================================

각 텐서에 양자화 매개변수를 지정하는 방식에도 차이가 있다.
Warboy는 activation 텐서에는 per-tensor 방식이, convolution의 weight와 bias는 per-channel 방식을 적용하고 있다.

per-tensor
--------------------------------------

* 텐서의 양자화 매개변수를 하나로 고정한다.
* 텐서 내의 모든 원소들이 같은 양자화 매개변수를 가지게 된다.

per-channel
--------------------------------------

* 텐서의 한 축에 대해 여러 개의 양자화 매개변수를 가진다.
* per-tensor 방식과 달리 각 원소 묶음마다 다양한 양자화 매개변수를 가질 수 있어 보다 정확하게 실수를 표현할 수 있다.
* 참고: https://arxiv.org/pdf/1806.08342v1.pdf


보정 (calibration)
======================================

양자화 과정 중 표현하고자 하는 실수의 범위를 결정하는 것은 중요한 단계이다.
이 실수의 범위를 계산하고 구하는 과정을 보정(calibration)이라고 한다.
이 과정을 통해 산출된 실수 범위를 보정 범위(calibration range)라고 한다.
보정 방식은 크게 두 가지로 분류된다.

PTQ (Post Training Quantization)
--------------------------------------

학습을 마친 원본 모델을 기반으로 activation, weight의 보정 범위를 구한다.
모델에 입력값을 넣어 실행하고 각 activation 텐서에서 사용되는 원소 값들을 기반으로 보정 범위를 구한다.
PTQ는 두 가지 방식으로 나누어진다.

* Post Training Dynamic Quantization

    * 모델을 실행하는 시점에 입력으로 들어온 값을 기반으로 보정 범위를 구한다.
    * 실행 시점 이전에는 보정 범위를 계산할 필요가 없다.
    * 실행 시점에 보정 범위를 계산하기 때문에 오버헤드가 발생한다.

* Post Training Static Quantization

    * 보정 범위를 실행 시점 이전에 미리 계산하고 모델에 정적으로 기록한다.
    * 모델에 이미 기록된 값을 사용하므로 실행 시점에는 오버헤드가 없다.
    * 보정 범위를 계산하기 위해 보정용 데이터 셋이 필요하다.


QAT (Quantization Aware Training)
--------------------------------------

모델을 학습하는 시점에 양자화를 고려해서 보정 범위를 계산한다.

(참고: https://arxiv.org/pdf/1712.05877.pdf)



FuriosaAI SDK의 보정 방식
======================================

FuriosaAI SDK는 기본적으로 Post Training Static Quantization을 보정 방식으로 사용하고 있다.
현재는 5개의 계산 방법을 제공하고 각각 대칭/비대칭 양자화를 지원한다.

.. list-table:: Calibration Method
   :header-rows: 1

   * - Method
     - Asymmetric
     - Symmetric
   * - MIN_MAX
     - MIN_MAX_ASYM
     - MIN_MAX_SYM
   * - ENTROPY
     - ENTROPY_ASYM
     - ENTROPY_SYM
   * - PERCENTILE
     - PERCENTILE_ASYM
     - PERCENTILE_SYM
   * - SQNR
     - SQNR_ASYM
     - SQNR_SYM
   * - MSE
     - MSE_ASYM
     - MSE_SYM


범위
--------------------------------------

보정 범위를 어떤 형태로 저장할지에 따라 두 가지로 나뉜다.

* 대칭형 (``SYM``)

    * Symmetric Quantization 방식으로 범위가 대칭형으로 정해진다.
    * 단, 범위 내의 값이 모두 양수일 경우 [-a, a]가 아닌 [0, a]로 산출된다.

        * 이를 통해 모든 원소의 값들이 양수임에도 불구하고 음수 영역으로 범위가 확장되어 실수 표현력이 저해되는 문제를 해결할 수 있다.

* 비대칭형 (``ASYM``)

    * Asymmetric Quantization 방식으로 범위가 비대칭형으로 정해진다.


산출 방법
--------------------------------------

보정 범위를 계산하는 방법을 5가지 제공하고 있다.
값 자체만 고려하여 계산하는 방식과, 값의 분포 즉 히스토그램을 바탕으로 계산하는 방식으로 나뉜다.

* 비 히스토그램 기반

    * ``MIN_MAX``

        * 텐서의 원소 중 최솟값과 최댓값을 보정 범위로 지정한다.
        * 분포에서 크게 벗어나 존재하는 원소 값(outlier)이 있을 경우 범위가 과도하게 넓게 잡히는 단점이 있다.

* 히스토그램 기반

    * ``ENTROPY``

        * 양자화 전의 분포와 양자화 후의 분포가 가장 유사한 보정 범위를 찾는다.
        * 원소 값들이 많이 분포되어 있는 곳을 최대한으로 표현한다.

    * ``PERCENTILE``

        * 원소 값 분포에서 비율을 계산하고 해당 퍼센티지를 포함할 수 있는 보정 범위를 찾는다.
        * outlier에 의해 발생하는 MIN_MAX의 단점을 완화시킬 수 있는 방법이다.

    * ``SQNR``: Signal-to-quantization-noise Ratio

        * 분포와 상관없이 양자화 후 다시 실수로 만들었을 때 오차가 작은 보정 범위를 찾는다.
        * 참고: https://arxiv.org/pdf/1511.06393.pdf

    * ``MSE``: Mean square quantization error

        * ``SQNR`` 과 같은 방식이나, 오차를 계산할 때 mean square를 이용한다.



FuriosaAI SDK의 Quantization 과정
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


ModelEditor
========================================

모델을 양자화하면 각 연산들의 입력과 출력은 정수 자료형으로 변경된다.
그러나 모델 자체의 입력과 출력 텐서 자료형은 여전히 실수 자료형으로 남아있다.
NPU에서 보다 원활하게 연산이 가속될 수 있도록 모델의 입력 또는 출력 텐서의 자료형을 변경할 수 있다.

다음의 API를 사용할 수 있다.

.. code-block:: python

    # to be update

위 API에 대한 자세한 설명은 성능 최적화 문서를 참고할 수 있다.


모델 양자화 APIs
========================================

SDK가 제공하는 API와 명령행 도구를 사용하여 ONNX 모델을 8bit 양자화 모델로 변환할 수 있다.
사용 방법은 아래에서 찾아볼 수 있다.

* `Python SDK 예제: 모델 생성 부터 인퍼런스 까지 <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/HowToUseFuriosaSDKFromStartToFinish.ipynb>`_
* `Python SDK Quantization 예제 <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/quantizers>`_
* `Python 레퍼런스 - furiosa.quantizer <https://furiosa-ai.github.io/docs/latest/en/api/python/furiosa.quantizer.html>`_
