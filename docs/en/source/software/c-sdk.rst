.. _CSDK:

*********************************************************
C SDK 설치 및 사용 가이드
*********************************************************

이 섹션은 C 언어를 이용해 FuriosaAI의 NPU 응용 애플리케이션을 작성할 때 필요한 가이드를
담고 있다. C SDK는 C ABI 기반의 정적 라이브러리 (static library) 와
C 헤더 (C header) 파일을 제공하며 이를 이용하여 C, C++ 응용 또는 C ABI를 지원하는
다른 언어로 응용을 작성할 수 있다.

:ref:`Python SDK <PythonSDK>` 에 비해 저수준 API를 제공하여
더 낮은 지연 시간과 더 높은 성능을 요구하는 경우나 Python 런타임을 사용할 수 없는 경우 사용할 수 있다.
또한, C SDK는 Python SDK와 유사한 블럭킹(blocking) 및 비동기(asynchnous) API를 제공한다.

C SDK 설치
===================================

C SDK 설치를 위한 최소 요구사항은 다음과 같다.

* Ubuntu 18.04 LTS (Debian buster) 또는 상위 버전
* 시스템의 관리자 권한 (root)
* :ref:`FuriosaAI SDK 필수 패키지 <RequiredPackages>`

또한, C SDK 설치 및 사용을 위해서는 :ref:`필수 패키지 설치 <RequiredPackages>`
가이드를 따라 드라이버, 펌웨어, 런타임 라이브러리를 반드시 설치해야 한다.
필수 패키지를 설치했다면 아래 방법으로 C SDK를 설치한다.

.. tabs::

  .. tab:: APT 서버를 이용한 설치

    FuriosaAI APT를 사용하기 위해서는 :ref:`SetupAptRepository` 을 따라
    서버 접속을 위한 인증 설정을 완료한다.

    .. code-block:: sh

      apt-get update && apt-get install -y furiosa-libnux-dev

  .. tab:: 다운로드 센터를 이용한 설치

    다운로드 센터에 로그인하여 아래 패키지들으 최신 버전을 선택하여 다운 받는다.

    * NPU C SDK 다운로드 (furiosa-libnux-dev-x.y.z-?.deb)

    .. code-block:: sh

      $ apt-get install -y ./furiosa-libnux-dev-x.y.z-?.deb


C SDK를 이용한 컴파일
===================================

.. warning::
  TODO

