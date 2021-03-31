.. Furiosa SDK Documents master file, created by
   sphinx-quickstart on Tue Mar 23 11:18:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FuriosaAI SDK 문서
=================================================

이 문서에서는 FuriosaAI SDK 설치 방법과 사용 방법을 제공한다.

.. note::

   FuriosaAI SDK는 명령 줄 도구, 런타임 라이브러리,
   Python 라이브러리를 포함한다. FuriosaAI NPU의 커널 드라이버, 펌웨어 및 런타임은
   FuriosaAI의 평가 프로그램 등록과 최종 사용자 라이센스 동의(End User License Agreement)에 따라
   배포되며, contact@furiosa.ai 로 문의하여 프로그램 다운로드 및 평가를 진행 할 수 있다.


SDK 설치
-------------------------------------------------
* :doc:`SDK 설치 사전 준비 (필수)</installation/prerequisites>` : SDK 설치를 위한 필수 준비 사항
* :doc:`NPU 커널 드라이버 및 펌웨어 설치 (필수)</installation/driver>` : NPU 를 구동하기 위한 커널 드라이버 및 펌웨어
* :doc:`NPU Runtime 설치 (필수)</installation/runtime>` : NPU 런타임 및 NPU 응용 프로그램 도구
* :doc:`NPU Python SDK 설치 (선택)</installation/python-sdk>` : NPU 사용을 위한 Python 라이브러리 및 명령행 도구
* :doc:`웹 서비스 API 키 설정 (선택)</installation/apikey>` : 웹 서비스로 제공하는 도구 사용을 위한 API 키 설정

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: SDK 설치

   /installation/prerequisites
   /installation/driver
   /installation/runtime
   /installation/python-sdk
   /installation/apikey

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 빠른 시작

   /quickstart/python-sdk
   /quickstart/cli

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 고급 주제

   /advanced/quantization
   /advanced/supported_operators

빠른 시작
-------------------
* :doc:`명령형 도구 빠르게 시작하기 </quickstart/cli>`
* :doc:`Python SDK 빠르게 시작하기 </quickstart/python-sdk>`

고급 주제
-------------------
* :doc:`모델 양자화 </advanced/quantization>`
* :doc:`지원 연산자 목록 </advanced/supported_operators>`
   

코드 예제
-------------------
* `Python SDK Quantization 예제 <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/furiosa-sdk-quantizer>`_
* `Python SDK Inference 예제 <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/furiosa-sdk-runtime>`_


기타 링크
-------------------
* `FuriosaAI 홈페이지 <https://furiosa.ai>`_
* `FuriosaAI SDK Github <https://github.com/furiosa-ai/furiosa-sdk>`_
* `C Language SDK 레퍼런스 <https://furiosa-ai.github.io/renegade-manual/references/nux/>`_
* `Python SDK 레퍼런스 <https://furiosa-ai.github.io/renegade-manual/references/python/>`_

