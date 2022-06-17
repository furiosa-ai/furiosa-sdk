.. _Profiling:

***********************************
성능 프로파일링
***********************************

많은 DNN 응용에서 낮은 지연시간과 높은 처리 성능은 중요한 요소이다.
성능 최적화를 위해서는 모델 개발자나 ML 엔지니어가 모델의 성능을 이해하고 병목지점을 분석할 수 있어야 한다.
이를 위해 FuriosaAI SDK는 프로파일링 도구를 제공한다.

트레이스 분석 (Trace Analysis)
---------------------------------------------------
트레이스 분석은 모델의 추론 작업을 실제로 실행 시키고 측정한 구간 별 실행 시간을 구조적 데이터(structured)로 제공한다.
또한 데이터를 크롬 웹브라우져(Chrome Web Browser)의 
`Trace Event Profiling Tool <https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/>`_ 
기능을 이용해 시각화 할 수 있다. 

트레이스 생성은 구간 별 시간을 측정하고 파일로 결과를 쓰기 때문에 작지만 실행 시간 오버헤드를 유발한다. 
따라서 기본 설정으로는 활성화 되어 있지 않으며 아래 두 가지 방법을 통해 트레이스를 생성할 수 있다.


환경 변수를 통한 트레이스 생성 활성화
============================================================
``FURIOSA_PROFILER_OUTPUT_PATH`` 에 트레이스 결과가 쓰여질 파일의 경로를 설정 하면
트레이스 생성을 활성화 할 수 있다. 이 방법은 작성된 코드를 전혀 변경하지 않고 
트레이스를 활성화할 수 있다는 장점이 있다. 반면, 측정하기 윈하는 구간이나 측정 연산의 카테고리를 더 세밀하게 설정 할 수 없다는 
한계가 있다.

.. code-block:: sh

    cd furiosa-sdk/examples/inferences
    export FURIOSA_PROFILER_OUTPUT_PATH=`pwd`/tracing.json
    ./image_classification.sh ../assets/images/car.jpeg

    ls -l ./tracing.json
    -rw-r--r--@ 1 furiosa  furiosa  605054 Jun  15 02:02 tracing.json


위와 같이 환경변수를 통해 설정하면 환경변수 값에 설정된 경로에 JSON 파일이 쓰여진다. 
크롬 브라우져에서 주소창에서 ``chrome://tracing`` 주소를 입력하면 트레이스 뷰어가
시작된다. 트레이스 뷰어에서 왼쪽 상단에 ``Load`` 버튼을 누르고 저장된 파일 (위의 예에서는 ``tracing.json``) 
파일을 선택하면 트레이스 결과를 볼 수 있다.

.. image:: ../../../imgs/tracing.png
  :alt: Tracing
  :class: with-shadow
  :align: center
  :width: 600


..
  for bottom margin of the above image

\

프로파일러 컨텍스트를 이용한 트레이스 생성 활성화
============================================================
Python 코드에서 프로파일러 컨텍스트(Profiler Context)를 정의하는 것으로도 트레이스 생성을 활성화할 수 있다.
이 방법은 환경 변수를 통한 트레이스 활성화 방법과 비교하여 다음과 같은 장점을 가진다.

* Python 인터프리터 또는 Jupyter Notebook와 같은 인터렉티브한 환경에서 바로 트레이스를 활성화 할 수 있다.
* 실행 구간에 레이블을 붙일 수 있다.
* 측정을 원하는 연산 카테고리의 선택적으로 측정할 수 있다.

.. code-block:: python
    
    from furiosa.runtime import session, tensor
    from furiosa.runtime.profiler import profile

    with open("mnist_trace.json", "w") as output:
        with profile(file=output) as profiler:
            with session.create("MNISTnet_uint8_quant_without_softmax.tflite") as sess:
                input_shape = sess.input(0)

                with profiler.record("warm up") as record:
                    for _ in range(0, 10):
                        sess.run(tensor.rand(input_shape))

                with profiler.record("trace") as record:
                    for _ in range(0, 10):
                        sess.run(tensor.rand(input_shape))



위는 프로파일링 컨텍스트를 활용한 코드 예제이다. 위 파이썬 코드가 실행되고 나면 `mnist_trace.json` 파일이 생성되며
트레이스 결과는 아래 그림과 같이 'warm up'과 'trace' 레이블이 붙는다.


.. image:: ../../../imgs/mnist_trace.png
  :alt: Tracing with Profiler Context
  :class: with-shadow
  :align: center
  :width: 600


\

