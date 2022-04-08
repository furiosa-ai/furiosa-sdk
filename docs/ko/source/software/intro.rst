**********************************
FuriosaAI 소프트웨어 스택 소개
**********************************

FuriosaAI는 NPU가 다양한 응용과 환경에서 사용될 수 있도록 다양한 소프트웨어 컴포넌트를
제공하고 있다. 이 장은 FuriosaAI가 제공하는 전반적인 소프트웨어 스택을 소개하고
각 컴포넌트의 역할과 컴포넌트 활용을 위한 설명서와 튜토리얼을 소개한다.

.. figure:: ../../../imgs/software_stack.jpg
  :alt: FuriosaAI Software Stack
  :class: with-shadow
  :width: 500px
  :align: center

위 그림은 FuriosaAI가 제공하는 소프트웨어 스택을 레이어에 따라 추상적으로 표현한 것이다.
가장 아래는 FuriosaAI의 1세대 NPU인 :ref:`IntroToWarboy` 가 있다.
아래는 주요한 컴포넌트에 대해 소개한다.

커널 장치 드라이버(Kernel Driver), 펌웨어(Firmware)
=============================================================
커널 장치 드라이버는 NPU 장치를 리눅스 운영체제 시스템이 인식하여
리눅스의 디바이스 파일로 인식하도록 하는 역할을 한다.
만약에 NPU 디바이스를 운영체제 차원에서 인식하지 않는다면
드라이버를 재설치하면 된다. 펌웨어는 리눅스 운영체제가 인식한 NPU 디바이스 파일을 기반으로
NPU 장치에 대한 저수준 API를 제공한다. 런타임과 컴파일러는 펌웨어가 제공하는
저수준 API를 이용하여 NPU를 제어하고 컴파일된 바이너리를 이용하여
NPU에서 추론을 위한 테스크를 실행하고 스케쥴링 한다.

커널 장치 드라이버와 펌웨어는 사용자가 직접 사용할 필요는 없으나
FuriosaAI SDK를 사용하기 위해서는 반드시 설치해야 한다.
설치 방법은 :ref:`RequiredPackages` 에서 찾을 수 있다.


컴파일러(Compiler)
====================================
컴파일러(Compiler)는 DNN 모델을 최적화하고 NPU에서 실행 가능한 코드를 생성하는 핵심적인 역할을 한다.
현재 컴파일러는 `TFLite <https://www.tensorflow.org/lite>`_, `ONNX <https://onnx.ai/>`_ 모델을
지원하며 다양한 최신 연구와 기법을 도입하여 모델을 최적화 하고 있다.

:ref:`Warboy <IntroToWarboy>` 와 함께 제공되는 컴파일러는 Vision 분야의 다양한 연산자(Operator)의
NPU 가속을 지원하며 가속을 지원하지 않는 연산자들은 CPU를 활용하도록 컴파일한다.
또한, Resnet50, SSD-Mobilenet, EfficientNet 과 같은 Vision 분야의
대표적인 모델을 잘 지원할 뿐만 아니라 사용자가 직접 설계하는 모델도
가속을 지원하는 연산자를 잘 활용하는 경우 NPU에 최적화된 코드를 생성할 수 있다.
참고로 NPU 가속을 지원하는 연산자는 :ref:`SupportedOperators` 에서 찾아볼 수 있다.

컴파일러는 런타임에 내장되어 제공되므로 사용자가 직접 설치할 필요는 없으며
Python/C SDK를 통해 세션(session)을 생성하는 과정에서 자동으로 사용되거나 :ref:`CompilerCli` 를 통해 사용할 수 있다.


런타임(Runtime)
=====================================
컴파일러가 생성한 실행 프로그램을 분석하고 프로그램에 기술된 DNN 모델 추론(Inference) 작업을 실제로 실행하는 역할을 한다.
컴파일 과정에서 DNN 모델의 추론은 최적화되고 NPU와 CPU에서 실행되는 다수의 작은 작업으로 분할된다.
런타임은 이 테스크를 머신의 자원을 균형있게 사용하고 워크로드에 맞도록 스케쥴링하고 NPU에서 실행되는 작업을 위해
펌웨어를 통해 NPU를 제어하는 역할을 수행한다.

런타임 기능은 아래 섹션에서 설명할 Python/C SDK를 통해 API로 제공되며
설치 방법은 :ref:`RequiredPackages` 에서 찾을 수 있다.

Python SDK와 C SDK
=====================================
Python과 C SDK는 런타임의 기능을 API로 Python과 C 라이브러리로 각각 제공하는 패키지이다.
지정한 모델이 지정한 디바이스를 사용하여 추론하도록 하는 세션(session)이라는 객체를 생성하는 API를 제공하고
API를 이용해 블럭킹, 비동기 방식으로 고성능 추론을 가능하게 한다.

NPU를 활용하는 응용프로그램이나 서비스를 작성해야 한다면
사용하는 응용의 프로그램 언어에 따라 둘 중 하나를 선택하여 설치하면 된다.
각 SDK의 설치와 사용법은 :ref:`PythonSDK` 와 :ref:`CSDK` 에서 찾을 수 있다.


모델 양자화 API (Quantizer)
=====================================
FuriosaAI SDK와 :ref:`Warboy <IntroToWarboy>` 는 8 비트 정수형 모델을 지원하며
실수형 데이터를 가중치로 가진 모델은 양자화(quantization)를 거쳐 :ref:`Warboy <IntroToWarboy>` 에서 사용할 수 있다.
이러한 양자화 과정을 간편하게 수행할 수 있도록 FuriosaAI SDK는 모델 양자화 API를 제공한다.
FuriosaAI SDK가 제공하는 양자화 API 대한 더 자세한 정보는 :ref:`ModelQuantization` 에서 찾을 수 있다.


모델 서버(Model Server)
=====================================
모델 서버는 DNN 모델을 GRPC나 REST API로 노출한다.
`TFLite <https://www.tensorflow.org/lite>`_, `ONNX <https://onnx.ai/>`_ 와 같은 모델 포맷은
모델 자체에 입력 텐서, 출력 텐서의 데이터 타입과 텐서 모양(tensor shape)이 기술되어 있는데 이 정보를 이용하여
보편적으로 사용되는 `Predict Protocol - Version 2 <https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md>`_
로 노출한다. 모델 서버를 사용하면 사용자는 직접 라이브러리와 Python/C SDK 통해 NPU에 접근할 필요가 없으며
원격 API를 통해 접근할 수 있다. 또한, 동일한 모델을 서빙하는 다수의 모델 서버를 사용하고
로드 밸런서를 이용하면 수평적 확장성 있는 서비스를 쉽게 구현할 수 있다.

모델 서버는 낮은 지연 시간(latency) 및 높은 동시성 처리 능력(throughput)을 요구하는데 이를 위해 런타임의 스케쥴링 기능을 활용한다.
모델 서버의 설치와 활용은 :ref:`ModelServing` 에서 찾을 수 있다.


Kubernetes 지원
======================================
컨테이너화된 워크로드와 서비스를 관리하는 플랫폼인 Kubernetes는 많은 기업에서 보편적으로 사용되고 있으며
FuriosaAI 소프트웨어 스택도 Kubernetes 네이티브 지원을 제공한다.

디바이스 플러그인(Kubernetes Device Plugin)은 Kubernetes 클러스터가 FuriosaAI의 NPU를 인식하고
NPU가 필요한 워크로드와 서비스를 위해 NPU를 스케쥴링할 수 있게 만든다.
이 기능은 Kubernetes와 같은 멀티테넌트(multi-tenant) 환경에서 다수의 워크로드가 NPU를 필요로 할 때
자원의 할당 문제를 돕고 한정된 NPU 자원을 효율적으로 활용할 수 있게 한다.

노드 레이블러(Kubernetes Node Labeller)는 Kubernetes에 참여하는 노드에 장착된 물리적인 NPU의 정보를
Kubernetes 노드 객체에 메타데이터로 추가하는 역할을 한다. 이 기능은 사용자가 Kubernetes API나 명령형 도구를
이용해 노드에 장착된 NPU 정보를 파악할 수 있게 하고
Pod의 ``spec.nodeSelector`` 나 ``spec.nodeAffinity`` 를 활용하여 특정 조건을 만족하는 노드에
워크로드를 배포할 수 있게 한다.

Kubernetes 환경에서 NPU 지원을 위한 설치 및 활용 방법은 :ref:`KubernetesIntegration` 페이지에서 찾을 수 있다.