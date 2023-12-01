.. _RequiredPackages:

***********************************************
드라이버, 펌웨어, 런타임 설치 가이드
***********************************************

이 장에서는 FuriosaAI가 제공하는 다양한 소프트웨어 컴포넌트를
사용하기 위해 반드시 설치해야 하는 패키지들의 설치 방법을 설명한다.
이 필수 패키지들은 커널 드라이버, 펌웨어, 런타임 라이브러리이며
APT 패키지 관리자를 통해 쉽게 설치할 수 있다.


.. note::

  FuriosaAI 평가 프로그램을 등록하고 나면 `FuriosaAI IAM <https://iam.furiosa.ai>`_ 에
  로그인하여 API 키를 발급받고 APT 서버를 이용할 수 있다.
  FuriosaAI 평가 프로그램을 등록은 contact@furiosa.ai 로 요청할 수 있다.

.. _MinimumRequirements:

SDK 설치를 위한 최소 요구사항
=====================================================================
* Ubuntu 20.04 LTS (Focal Fossa) 또는 Debian bullseye
  또는 상위 버전
* 시스템의 관리자 권한 (root)
* 인터넷 접근이 가능한 네트워크


.. _SetupAptRepository:

APT 서버 설정
=====================================================================

FuriosaAI에서 제공하는 APT 서버를 사용하려면 아래 설명에 따라 APT 서버를
Ubuntu 또는 Debian 리눅스에 설정한다.


1. HTTPS 기반의 APT 서버 접근을 위해 필요 패키지를 설치 한다.

.. code-block:: sh

  sudo apt update
  sudo apt install -y ca-certificates apt-transport-https gnupg wget

2. FuriosaAI의 공개 Signing 키를 등록 한다.

.. code-block:: sh

  mkdir -p /etc/apt/keyrings && \
  wget -q -O- https://archive.furiosa.ai/furiosa-apt-key.gpg \
  | gpg --dearmor \
  | sudo tee /etc/apt/keyrings/furiosa-apt-key.gpg > /dev/null

3. FuriosaAI 개발자 센터에서 API 키를 발급하고 발급한 API 키를 아래와 같이 설정한다.


.. code-block:: sh

  sudo tee -a /etc/apt/auth.conf.d/furiosa.conf > /dev/null <<EOT
    machine archive.furiosa.ai
    login [KEY (ID)]
    password [PASSWORD]
  EOT

  sudo chmod 400 /etc/apt/auth.conf.d/furiosa.conf


4. 리눅스 배포판 버전에 따른 탭을 선택하여 설명을 따라 APT 서버를 설정한다.


.. tabs::

  .. tab:: Ubuntu 20.04 (Debian Bullseye)

      아래 커맨드를 통해 APT 서버를 등록합니다.

      .. code-block:: sh

        sudo tee -a /etc/apt/sources.list.d/furiosa.list <<EOT
        deb [arch=amd64 signed-by=/etc/apt/keyrings/furiosa-apt-key.gpg] https://archive.furiosa.ai/ubuntu focal restricted
        EOT

  .. tab:: Ubuntu 22.04 (Debian Bookworm)

      아래 커맨드를 통해 APT 서버를 등록합니다.

      .. code-block:: sh

        sudo tee -a /etc/apt/sources.list.d/furiosa.list <<EOT
        deb [arch=amd64 signed-by=/etc/apt/keyrings/furiosa-apt-key.gpg] https://archive.furiosa.ai/ubuntu jammy restricted
        EOT



.. _InstallLinuxPackages:

필수 패키지 설치
=====================================================================

위의 설명에 따라 APT 서버를 등록했거나 다운로드 사이트에 가입했다면 필수 패키지인
NPU 커널 드라이버, 펌웨어, 런타임을 설치할 수 있다.

.. tabs::

  .. tab:: APT 서버를 이용한 설치

    아래 패키지를 설치하면 의존된 패키지들은 자동으로 설치된다.

    .. code-block:: sh

      sudo apt-get update && sudo apt-get install -y furiosa-driver-warboy furiosa-libnux

  .. .. tab:: 다운로드 센터를 이용한 설치

  ..   아래 패키지들의 최신 버전을 선택하여 다운 받아 명령에 쓰여진 순서대로 설치한다.
  ..   ``x.y.z-?`` 버전 부분은 다운받은 파일의 버전에 맞게 변경한다.

  ..   * NPU Driver (furiosa-driver-warboy)
  ..   * Hardware Abstraction Layer (furiosa-libhal)
  ..   * Runtime library  (furiosa-libnux)
  ..   * Onnxruntime  (libonnxruntime)

  ..   .. code-block:: sh

  ..     sudo apt-get install -y ./furiosa-driver-warboy-x.y.z-?.deb
  ..     sudo apt-get install -y ./furiosa-libhal-warboy-x.y.z-?.deb
  ..     sudo apt-get install -y ./libonnxruntime-x.y.z-?.deb
  ..     sudo apt-get install -y ./furiosa-libnux-x.y.z-?.deb


.. _AddUserToFuriosaGroup:

유저를 ``furiosa`` 그룹에 추가
------------------------------

리눅스는 다중 사용자 운영체제로 소유자(owner)와 그룹에 속한 사용자만 파일과 장치 접근이 허용된다. NPU 디바이스 드라이버도 설치 시 자동으로 ``furiosa`` 그룹을 생성하고 이 그룹에 속한 사용자만 접근을 허용하고 있다.
따라서 NPU를 사용하려는 유저는 자신을 ``furiosa`` 그룹에 추가해야 한다. 이를 위해 다음 명령어를 사용할 수 있다.

.. code-block:: sh

  sudo usermod -aG furiosa <username>


위 명령어에서 <username> 부분에 ``furiosa`` 그룹에 추가하려는 사용자의 이름을 입력하면 된다.
예를 들어, 현재 로그인된 사용자를 ``furiosa`` 그룹에 추가하려면, 다음 명령어를 입력한다.

.. code-block:: sh

  sudo usermod -aG furiosa $USER


그룹 권한을 활성화하기 위해서는 로그아웃한 후 다시 로그인해야 한다.


.. _HoldingAptVersion:

설치된 버전 고정 및 해제
------------------------------

패키지 설치 이후 안정적인 운영환경을 유지하기 위해 설치된 패키지의 버전을 고정할 필요가 있다.
아래 명령어를 이용하면 현재 설치된 버전을 고정 할 수 있다.

.. code-block:: sh

  sudo apt-mark hold furiosa-driver-warboy furiosa-libhal-warboy furiosa-libnux libonnxruntime


고정된 패키지 버전을 해제하여 업데이트 하기 위해서는 ``apt-mark unhold``
명령과 함께 원하는 패키지를 지정한다. 이때 패키지 이름을 적어 선택적으로 특정 패키지의 버전 고정을 해제할 수 있으며
이미 고정된 패키지 정보를 보기 위해서는 ``apt-mark showhold`` 명령을 사용한다.

.. code-block:: sh

  sudo apt-mark unhold furiosa-driver-warboy furiosa-libhal-warboy furiosa-libnux libonnxruntime


.. _InstallSpecificVersion:

특정 버전 설치 방법
------------------------------

특정 버전을 지정하여 설치해야 하는 경우 아래와 같이 버전을 지정하여 설치한다.

1. ``apt list`` 로 설치 가능한 버전을 확인한다.

.. code-block:: sh

  sudo apt list -a furiosa-libnux


2. ``apt-get install`` 명령에 옵션으로 패키지 이름과 버전을 지정한다.

.. code-block:: sh

  sudo apt-get install -y furiosa-libnux=0.9.1-?


.. _UpgradeFirmware:

NPU 펌웨어 업데이트
=====================================================================

APT 저장소를 통해 NPU 장치의 펌웨어 관리 도구를 설치하고 업데이트할 수 있다.

1. ``apt list`` 로 설치 가능한 버전을 확인한다.

    .. code-block:: sh

        sudo apt list -a 'furiosa-firmware-*'

        Listing... Done
        furiosa-firmware-image/focal 1.7.1 amd64
        furiosa-firmware-image/focal 1.7.0 amd64
        furiosa-firmware-image/focal 1.5.0 amd64
        furiosa-firmware-image/focal 1.4.0 amd64
        furiosa-firmware-image/focal 1.2.0 amd64

        furiosa-firmware-tools/focal 1.5.0-2 amd64
        furiosa-firmware-tools/focal 1.4.0-2 amd64
        furiosa-firmware-tools/focal 1.3.0-2 amd64
        furiosa-firmware-tools/focal 1.2.0-2 amd64


2. ``apt-get install`` 명령에 옵션으로 패키지 이름과 버전을 지정한다. 버전을 명시하지 않을 경우 최신 버전이 설치된다.

    .. code-block:: sh

        sudo apt-get install furiosa-firmware-tools

  또는

    .. code-block:: sh

        sudo apt-get install furiosa-firmware-tools furiosa-firmware-image=1.7.1


.. warning::

    컨테이너 환경에서의 펌웨어 업데이트는 정의되어 있지 않은 행동을 유발한다.
    따라서 이 작업은 반드시 호스트 머신에서 실행되어야 하므로 주의를 요한다.
