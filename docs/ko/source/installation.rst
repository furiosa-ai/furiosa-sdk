**********************************
FuriosaAI SDK 설치
**********************************

FuriosaAI NPU 사용을 위해서는 SDK 설치가 필요하며
이 페이지에는 FuriosaAI SDK 설치 방법을 안내한다.

.. note::

  FuriosaAI SDK의 일부 소프트웨어 (커널 드라이버, 펌웨어, 런타임)는
  FuriosaAI SDK 및 NPU 평가 프로그램을 통해 별도의 채널로 전달 받아야 한다.
  FuriosaAI SDK 및 NPU 평가 프로그램 등록은 contact@furiosa.ai 로 신청할 수 있다.


모든 SDK 설치를 위한 최소 요구사항은 다음과 같다.

  * 시스템의 관리자 권한(root)이 필요하다.
  * Ubuntu 18.04 LTS (Bionic Beaver) 또는 Debian buster
    또는 그 상위 버전이 필요하다.
  * 인터넷 네트워크에 연결되어 있어야 한다.


커널 드라이버 및 펌웨어는 모든 상황에서 필수적으로 설치해야 하며
나머지 컴포넌트는 필요에 따라 선택적으로 설치하면 된다.
아래 목차의 설명을 읽고 필요한 SDK 컴포넌트를 설치하자.

  * :doc:`FuriosaAI NPU 드라이버 및 펌웨어 설치<installation/driver>` : 시스템이 NPU를 인식할 수 있도록 커널 드라이버와 펌웨어를 설치한다. 다른 과정 진행을 위해 반드시 필요한 설치이다.
  * :doc:`FuriosaAI SDK Runtime 설치<installation/runtime>` : 로컬 시스템에서 NPU 응용 프로그램 개발이 필요하거나 실행이 필요할 때 설치한다.
  * :doc:`FuriosaAI Python SDK 설치<installation/python-sdk>` : 명령형 커맨드 도구(command line tool), Python에서 런타임 사용을 위한 라이브러리, 웹 서비스 기반 도구의 API 호출을 위해 설치한다.
  * :doc:`FuriosaAI API 키 설정<installation/apikey>` : 웹 서비스 기반 툴체인 및 도구를 사용하기 위해 필요하다.


.. toctree::
  :hidden:  

  installation/driver
  installation/runtime
  installation/python-sdk
  installation/apikey
  