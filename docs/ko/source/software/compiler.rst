.. _Compiler:

****************************************
컴파일러
****************************************
FuriosaAI의 컴파일러는 `TFLite <https://www.tensorflow.org/lite>`_ 와 `Onnx <https://onnx.ai/>`_
형식의 모델을 컴파일하여 FuriosaAI NPU와 호스트 머신의 자원(CPU, 메모리)을 이용해 추론(inference)을
실행하는 프로그램을 생성한다. 이 과정에서 모델을 연산자 단위로 분석하고 최적화하여 최대한 NPU 가속와 호스트 자원을
잘 활용할 수 있도록 프로그램을 생성한다. 따라서, 기존에 알려진 모델이 아니라도 지원되는
연산자를 잘 활용하면 NPU에 최적화된 모델을 설계할 수 있다.
NPU 가속이 지원되는 연산자 목록은 :ref:`SupportedOperators` 에서 찾을 수 있다.

.. _CompilerCli:

``furiosa compile``
-------------------------------------------------
컴파일러는 추론 API의 Session을 초기화 하는 과정에서 모델과 NPU를 초기화할 때
자동으로 호출되어 사용되는 것이 가장 일반적인 사용 방법이다.
그러나 쉘에서 명령행 도구인 ``furiosa compile`` 이용해 직접 모델을 컴파일하여 프로그램을 생성해볼 수 있다.
``furiosa compile`` 명령은 :ref:`PythonSDK` 를 설치하면 사용 가능해진다.

명령의 인자는 다음과 같다. ``MODEL_PATH`` 는
`TFLite <https://www.tensorflow.org/lite>`_ 나 `Onnx <https://onnx.ai/>`_ 파일의 경로이다.

.. code-block:: sh

  furiosa compile MODEL_PATH [-o OUTPUT] [--target-npu TARGET_NPU] [--batch BATCH_SIZE]

`-o OUTPUT` 은 생략 가능한 옵션이며 지정한다면 출력되는 파일 이름을 지정할 수 있다.
생략했을 때 기본 출력 파일 이름은 ``output.enf`` 이다. 여기서 enf는 Executable NPU Format의 약어이다.
따라서, 예를 들면 아래와 같이 실행하면 기본으로 ``output.enf`` 파일을 생성한다.

.. code-block:: sh

  furiosa compile foo.onnx

아래와 같이 직접 출력 파일 이름을 지정하면 ``foo.enf`` 파일로 생성된다.

.. code-block::

  furiosa compile foo.onnx -o foo.enf

``--target-npu`` 는 생성한 바이너리가 목표로하는 NPU를 지정하게 한다.

.. list-table:: Target NPUs
   :widths: 50 50 50
   :header-rows: 1

   * - NPU Family
     - Number of PEs
     - Value
   * - Warboy
     - 1
     - warboy
   * - Warboy
     - 2
     - warboy-2pe

생성한 프로그램이 동작할 NPU가 1개의 PE를 독립적으로 사용하는 Warboy라면 아래와 같이 명령을 실행하면 된다.

.. code-block::

  furiosa compile foo.onnx --target-npu warboy

2개의 PE (Processing Element)를 Fusing 해서 사용하는 경우는 아래와 같이 실행한다.

.. code-block::

  furiosa compile foo.onnx --target-npu warboy-2pe

``--batch-size`` 옵션은 추론 API를 통해 추론을 실행할 때
입력으로 전달할 샘플의 개수인 `배치 크기` 를 지정하게 한다.
배치 크기가 크면 일반적으로 한번에 많은 데이터를 넣고 실행하므로
NPU의 활용도를 높일 수 있고 추론을 실행하는 과정을 공유하므로 더 효율적일 수 있다.
그러나 NPU에 더 많은 메모리가 필요하게 되어 필요한 메모리 사이즈가 NPU의 DRAM 크기를 초과하면
오히려 호스트(Host)와 NPU간에 메모리 I/O 비용이 커져 큰 성능 저하가 일어날 수 있다.
기본 값은 1이며 적절한 설정은 일반적으로 실험을 통해 찾을 수 있다.
참고로, `MLPerf™ Inference Edge v2.0 <https://mlcommons.org/en/inference-edge-20/>`_
벤치마크에 포함된 일부 모델들의 최적 배치 크기는 다음과 같다.

.. list-table:: Optimal Batch Size for Well-known Models
   :widths: 50 50
   :header-rows: 1

   * - Model
     - Optimal Batch
   * - SSD-MobileNets-v1
     - 2
   * - Resnet50-v1.5
     - 1
   * - SSD-ResNet34
     - 1


원하는 배치 크기가 2인 경우는 아래와 같이 명령을 실행하면 된다.

.. code-block::

  furiosa compile foo.onnx --batch-size 2


ENF 파일의 활용
---------------------------------
FuriosaAI 컴파일러가 컴파일 과정을 마치고 최종적으로 생성해내는 출력물이
ENF (Executable NPU Format) 형식의 데이터이다.
일반적으로, 컴파일 과정은 모델에 따라서 수 초에서 수 분까지 걸리는데
ENF 파일을 한번 생성하여 재사용하면 컴파일을 과정을 생략할 수 있다.

예를 들면, 아래 처럼 :ref:`PythonSDK <PythonSDK>` 를 사용할 때 ``session.create()``
함수에 인자로 ENF 파일을 전달하면 컴파일 과정을 거치지 않고 즉각적으로
``Session`` 객체를 생성한다.

.. code-block:: python

  from furiosa.runtime import session
  sess = session.create("foo.enf")