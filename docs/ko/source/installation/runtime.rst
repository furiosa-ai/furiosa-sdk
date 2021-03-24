**********************************
FuriosaAI Runtime 라이브러리 설치
**********************************

FuriosaAI Runtime 라이브러리는 로컬 머신에서 
FuriosaAI NPU의 응용 프로그램을 실행하거나 개발할 때 필요합니다.

평가 프로그램 채널을 통해 아래 두 파일을 다운받으세요. ``x.y.z`` 은 SDK 릴리즈 버전이며 
일반적인 경우 가장 최신 버전을 다운로드 받으시면 됩니다.

  * ``furiosa-libnux-x.y.z_amd64.deb``:  Runtime 라이브러리
  * ``furiosa-libnux-dev-x.y.z_amd64.deb``: Runtime 개발 환경 라이브러리(C 헤더 파일, 정적 라이브러리)

``x.y.z`` 를 다운받은 SDK 버전에 맞게 변경하고 아래 명령으로 시스템에 설치해주세요.

.. code-block::

  $ wget https://github.com/hyunsik/onnxruntime/releases/download/v1.5.2/libonnxruntime-1.5.2.deb
  $ apt-get install -y ./libonnxruntime-1.5.2.deb
  $ apt-get install -y ./furiosa-libnux-x.y.z_amd64.deb
  $ apt-get install -y ./furiosa-libnux-dev-x.y.z-dev_amd64.deb