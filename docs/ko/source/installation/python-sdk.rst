**********************************
FuriosaAI Python SDK 설치
**********************************

FuriosaAI NPU Python SDK 는 모델을 NPU 에서 가속시키기 위한 각종 인터페이스를 제공하며, Python 라이브러리 및
명령줄 도구(command line tool)를 포함한다.

요구 사항
----------------------------------------
* :doc:`./driver`
* :doc:`./runtime`
* Python 3.7 또는 상위 버전 (Python 환경 구성은 :any:`SetupPython` 를 참고)
* pip 최신 버전 (다음 방법으로 pip를 최신버전으로 업그레이드)

  .. code-block::

        $ pip3 install --upgrade pip

설치
----------------------------------------

FuriosaAI Python SDK는 PyPi를 활용하여 설치할 수 있도록 PyPi 저장소에 준비되어 있다.
Python 라이브러리 인터페이스 이외에도 명령줄 도구 및 다양한 기능을 사용할 수 있고, 다음의 예시대로 설치 가능하다.

.. code-block:: sh

  # FuriosaAI NPU Python SDK 설치, Python 인터페이스 사용 가능, e.g. `import furiosa`
  pip install --upgrade furiosa-sdk~=0.1.0
  # 부가 도구 설치, 자세한 목록은 아래 참조
  pip install --upgrade furiosa-sdk[runtime,quantizer]~=0.1.0
  # 부가 도구 전체 설치
  pip install --upgrade furiosa-sdk[full]~=0.1.0

PIP 커맨드를 이용하여 다음 부가 패키지를 설치할 수 있다.

* ``cli``: 명령줄 도구 설치, 사용 방법은 :doc:`/quickstart/cli` 를 참고

  .. code-block::

    pip install --upgrade furiosa-sdk[cli]

* ``runtime``: FuriosaAI NPU Runtime 을 사용하여 NPU 위에서 모델을 가속시키기 위한 각종 라이브러리 설치, **NPU 위에서 모델 가속을 위해 필수**

  .. code-block::

    pip install --upgrade furiosa-sdk[runtime]~=0.1.0

* ``quantizer``: 모델의 양자화 도구 설치 (:doc:`/advanced/quantization` 참고)

  .. code-block::

    pip install --upgrade furiosa-sdk[quantizer]~=0.1.0

* ``validator``: 모델 분석 도구 설치, 해당 모델이 NPU 위에서 가속되기 위해 양자화, 컴파일이 잘 수행되는지 확인하는 도구를 포함

  .. code-block::

    pip install --upgrade furiosa-sdk[quantizer,runtime,validator,cli]~=0.1.0


예를 들어 모델 추론을 위한 개발환경과 모델 양자화 도구가 필요한 경우 아래와 같이 설치한다.

.. code-block:: sh

  pip install --upgrade furiosa-sdk[runtime,quantizer]~=0.1.0


Jupyter Notebook 사용 안내
----------------------------------------

Jupyter Notebook을 사용하는 경우
FuriosaAI Python SDK와 다양한 Python 에코시스템의 다양한
라이브러리를 편하게 사용할 수 있다.

위 설명에 따라 Python SDK를 이미 설치했다면 
아래와 같이 pip를 이용해 Jupyter notebook을 간단히 설치해 사용할 수 있다.
Jupyter notebook은 아주 다양한 의존된 패키지를 설치하기 때문에
:ref:`CondaInstall` 을 권장한다.

.. code-block:: sh
  
  $ pip install jupyterlab
  $ jupyter-notebook