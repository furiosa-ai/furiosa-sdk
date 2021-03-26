**********************************************
FuriosaAI NPU Python SDK 빠르게 시작하기
**********************************************

FuriosaAI NPU Python SDK 라이브러리를 사용하면 NPU 를 사용하는 프로그램을 Python 으로 쉽게 작성할 수 있다.

설치
================================

요구사항

  * Python 3.7+

    * :doc:`/installation/python-sdk-setup-python`

  * :doc:`/installation/driver`

  * :doc:`/installation/runtime`


위 요구 사항을 만족하는 경우 Python 환경이 준비된 경우 PyPi 를 통해 설치할 수 있다.

.. code-block:: sh

  pip install --upgrade furiosa-sdk[runtime]~=0.1.0

Python 환경 구성과 다양한 설치 옵션에 대해서는 :doc:`/installation/python-sdk` 에서 자세히 살펴볼 수 있다.


간단한 Python 코드 실행
=================================
설치가 완료되면 Python 코드를 실행하여 버전 정보를 출력하여 패키지가 잘 설치되었는지 확인해볼 수 있다.

.. code-block::
  
  >> from furiosa import runtime
  INFO:furiosa_sdk_runtime._api.v1:successfully loaded dynamic library libnux.so.1.0.0