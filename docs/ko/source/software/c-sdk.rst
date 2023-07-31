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
또한, C SDK는 Python SDK와 유사한 블럭킹(blocking) 및 비동기(asynchronous) API를 제공한다.

.. warning::

  ``furiosa-libnux-dev`` 패키지와 현재 C API는 향후 릴리즈에서 제거될 예정이다.

  현재 C API를 대체하는 새로운 C API가 향후 릴리즈에 제공될 예정이다. 새 C API는 0.10.0 부터 제공되는
  차세대 런타임인 FuriosaRT 를 기반으로 하며 더 많은 기능을 제공할 예정이다.


C SDK 설치
===================================

C SDK 설치를 위한 최소 요구사항은 다음과 같다.

* Ubuntu 20.04 LTS (Debian bullseye) 또는 상위 버전
* 시스템의 관리자 권한 (root)
* :ref:`FuriosaAI SDK 필수 패키지 <RequiredPackages>`

또한, C 언어 SDK 설치 및 사용을 위해서는 :ref:`필수 패키지 설치 <RequiredPackages>`
가이드를 따라 드라이버, 펌웨어, 런타임 라이브러리를 반드시 설치해야 한다.
필수 패키지를 설치했다면 아래 방법으로 C SDK를 설치한다.

.. tabs::

  .. tab:: APT 서버를 이용한 설치

    FuriosaAI APT를 사용하기 위해서는 :ref:`SetupAptRepository` 을 따라
    서버 접속을 위한 인증 설정을 완료한다.

    .. code-block:: sh

      apt-get update && apt-get install -y furiosa-libnux-dev

  .. .. tab:: 다운로드 센터를 이용한 설치

  ..   다운로드 센터에 로그인하여 아래 패키지들의 최신 버전을 선택하여 다운 받는다.

  ..   * NPU C SDK 다운로드 (furiosa-libnux-dev-x.y.z-?.deb)

  ..   .. code-block:: sh

  ..     $ apt-get install -y ./furiosa-libnux-dev-x.y.z-?.deb


C SDK를 이용한 컴파일
===================================
위와 같이 패키지를 설치하면 C 언어 SDK를 이용하여 컴파일할 수 있다.
C 헤더 파일은 ``/usr/include/furiosa`` 디렉토리에, 정적 라이브러리는
``/usr/lib/x86_64-linux-gnu`` 디렉토리에 위치한다. 위 경로들은 시스템에서 헤더와 라이브러리를 찾는
기본 경로에 포함되어 있으므로 헤더 파일을 포함하는 경우 ``#include <furiosa/nux.h>`` 와 같이 사용할 수 있다.
라이브러리를 링크하기 위해서는 ``-lnux`` 옵션만 추가하면 된다. 예를 들어, 다음과 같이 컴파일할 수 있다.

.. code-block:: sh

  gcc example.c -lnux


또한, `C Language SDK 레퍼런스 <https://furiosa-ai.github.io/docs/v0.10.0/en/api/c/index.html>`_ 에서
예제 코드와 API 문서를 확인할 수 있다.
