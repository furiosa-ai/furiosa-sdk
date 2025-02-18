****************************************************
FuriosaAI Warboy 및 SDK |release| 문서
****************************************************

이 문서는 FuriosaAI Warbo 소개와 NPU 활용에 필요한 Software 설치 및 사용 방법을 제공한다.

.. note::

   FuriosaAI의 소프트웨어는 커널 드라이버, 펌웨어 및 런타임 및 C SDK, Python SDK 및 명령행 도구를 포함한다.
   FuriosaAI의 소프트웨어는 평가 프로그램(Early Access Program) 등록 후 최종 사용자
   라이센스 동의(End User License Agreement)에 따라 배포되며, 현재는 contact@furiosa.ai 로 문의하여
   평가 프로그램을 시작할 수 있다.


FuriosaAI Warboy
-------------------------------------------------
* :doc:`FurisaAI Warboy </npu/warboy>`: 하드웨어 사양, 성능, 가속 연산자 소개

FuriosaAI Software
-------------------------------------------------
* :doc:`FuriosaAI 소프트웨어 스택 소개</software/intro>`
* :doc:`드라이버, 펌웨어, 런타임 설치 가이드 </software/installation>`
* :doc:`Python SDK 설치 및 사용 가이드 </software/python-sdk>`
* :doc:`C SDK 설치 및 사용 가이드 </software/c-sdk>`
* :doc:`SDK 명령행 도구 </software/cli>`
* :doc:`컴파일러 </software/compiler>`
* :doc:`모델 양자화 (Model quantization) </software/quantization>`
* `FuriosaAI Model Zoo <https://furiosa-ai.github.io/furiosa-models/latest/>`_
* :doc:`Kubernetes 지원 </software/kubernetes_support>`
* :doc:`가상 머신 환경 지원 </software/vm_support>`

FuriosaAI SDK 튜토리얼 및 코드 예제
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `Python SDK 시작하기 <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/GettingStartedWithPythonSDK.ipynb>`_
* `고급 추론 API <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/AdvancedTopicsInInferenceAPIs.ipynb>`_
* `CPU 기반 추론과 정확도 비교하기 예제 <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/InferenceAccuracyCheck.ipynb>`_
* `이미지 분류(Image Classification) 모델 추론 <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/Image_Classification.ipynb>`_
* `객체 탐지 (SSD Object Detection) 모델 추론 <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/SSD_Object_Detection.ipynb>`_
* `기타 Python 코드 예제 <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/inferences>`_

서빙, 배포, MLOps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :doc:`모델 서버 (서빙 프레임워크) </software/serving>`
* :doc:`Kubernetes 지원 </software/kubernetes_support>`

레퍼런스 문서
-------------------------------------------------
* `C Language SDK 레퍼런스 <https://furiosa-ai.github.io/docs/v0.5.0/en/api/c/index.html>`_
* `Python SDK 레퍼런스 <https://furiosa-ai.github.io/docs/v0.5.0/en/api/python/modules.html>`_

기타 링크
--------------------------------------------------
* `FuriosaAI 홈페이지 <https://furiosa.ai>`_
* `Furiosa SDK 사용자 게시판 <https://furiosa-ai.discourse.group/>`_
* `FuriosaAI 고객지원 센터 <https://furiosa-ai.atlassian.net/servicedesk/customer/portals/>`_
* :ref:`BugReport`


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: FuriosaAI NPU

   /npu/warboy

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: FuriosaAI 소프트웨어

   /software/intro
   /software/installation
   /software/python-sdk
   /software/c-sdk
   /software/cli
   /software/compiler
   /software/quantization
   /software/performance
   /software/profiler
   FuriosaAI Model Zoo <https://furiosa-ai.github.io/furiosa-models/latest/>
   /software/serving
   /software/kubernetes_support
   /software/vm_support
   /software/tutorials
   /software/references


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: SDK 릴리즈 노트

   /releases/0.10.0.rst
   /releases/0.9.0.rst
   /releases/0.8.0.rst
   /releases/0.7.0.rst
   /releases/0.6.0.rst
   /releases/0.5.0.rst


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 고객 지원

   Furiosa SDK 사용자 게시판 <https://furiosa-ai.discourse.group/>
   FuriosaAI 고객지원 센터 <https://furiosa-ai.atlassian.net/servicedesk/customer/portals/>
   /customer-support/bugs

..
   다운로드 센터 <https://developer.furiosa.ai/downloads/>


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 기존 버전 문서

   0.9.0 문서 <https://furiosa-ai.github.io/docs/v0.9.0/ko/>
   0.8.0 문서 <https://furiosa-ai.github.io/docs/v0.8.0/ko/>
   0.7.0 문서 <https://furiosa-ai.github.io/docs/v0.7.0/ko/>
   0.6.3 문서 <https://furiosa-ai.github.io/docs/v0.6.3/ko/>
   0.6.0 문서 <https://furiosa-ai.github.io/docs/v0.6.0/ko/>
   0.5.0 문서 <https://furiosa-ai.github.io/docs/v0.5.0/ko/>
   0.2.0 문서 <https://furiosa-ai.github.io/docs/v0.2.0/ko/>
