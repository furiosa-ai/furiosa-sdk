.. _Profiling:

***********************************
프로파일링(Profiling) 사용 가이드
***********************************

다수의 DNN 응용에서 낮은 지연시간과 높은 처리 성능은 중요한 요소이다.
성능 최적화를 위해서는 모델 개발자나 ML 엔지니어가 모델의 성능을 이해하고 병목지점을 분석할 수 있어야 한다.
이를 위해 FuriosaAI SDK는 프로파일링 도구를 제공한다.

트레이스 분석 (Trace Analysis)
---------------------------------------------------
트레이스 분석은 모델의 추론 작업을 실제로 실행 시키고 측정한 구간 별 실행 시간을 구조적 데이터(structured)로 제공한다.
또한 데이터를 크롬 웹브라우져(Chrome Web Browser)를 이용해 시각화 할 수 있다. 

트레이스 생성은 구간 별 시간을 측정하고 파일로 결과를 쓰기 때문에 실행 시간에
약간의 오버헤드를 야기한다. 따라서 기본 설정으로는 활성화 되어 있지 않으며 아래 두 가지 방법을 통해 트레이스를 생성할 수 있다.


환경 변수를 통한 트레이스 생성 활성화
===========================================
환경 변수를 지정하여 트레이스 생성을 활성화 할 수 있다. 
이 방법은 이 기능은 이미 작성된 코드를 전혀 변경하지 트레이스를 활성화할 수 있다는 장점이 있는 반면
측정하기를 윈하는 구간이나 연산의 카테고리를 더 세밀하게 설정 할 수 없다는 한계가 있다.

활성화 방법은 환경변수 ``FURIOSA_PROFILER_OUTPUT_PATH`` 에 트레이스 결과가 쓰여질 파일의 패스를 설정하는 것 이다.

.. code-block:: sh

    cd furiosa-sdk/examples/inferences
    export FURIOSA_PROFILER_OUTPUT_PATH=`pwd`/trace.json
    ./image_classification.sh ../assets/images/car.jpeg

    ls -l ./traces.json

`chrome://trace`_ 를 열어서 상단 왼쪽 위에 ``Load`` 버튼을 누르고 ``trace.json`` 파일을 선택하면
트레이스 결과를 볼 수 있다.


프로파일러 컨텍스트를 이용한 트레이스 생성 활성화
===========================================
Python 코드에 프로파일러 컨텍스트를 간단히 정의하는 것 만으로 트레이스 생성을 활성화할 수 있다.
이 방법의 환경 변수를 통한 트레이스 활성화 방법에 비해 다음과 같은 장점을 가진다.

* Python 인터프리터 또는 Jupyter Notebook를 사용 중에 트레이스를 활성화 할 수 있다.
* 실행 구간에 레이블을 붙일 수 있다.
* 측정을 원하는 카테고리의 구간만 선택적으로 측정할 수 있다.

이 방법의 주요한 사용 예는 모델을 개발하던 상황에서 실행시간에 모델 추론에 대한 트레이스를 바로 생성할 수 있다는 것이다.

.. code-block:: python
    
    from furiosa.runtime import profiler, session, tensor
    
    with profiler.create("..") as p:        
        with session.create("...") as sess:
            input_shape = sess.input(0)

            with p.record("warm up") as record:
                for _ in range(0, 10)
                    sess.run(tensor.rand(input_shape))

            with p.record("trace") as record:
                for _ in range(0, 10)
                    sess.run(tensor.rand(input_shape))

    # Here, trace.json has been written. You can access the file


