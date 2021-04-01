**********************************
NPU Runtime 설치
**********************************

FuriosaAI NPU Runtime 는 딥 러닝 모델을 NPU 위에서 가속하기 위한 NPU 런타임 및 이를 실행시키기 위한 인터페이스,
그리고 NPU 응용 프로그램 개발에 필요한 각종 도구를 포함한다.
아래 두 패키지를 설치하여 FuriosaAI NPU Runtime 을 사용할 수 있으며,
deb 패키지 구성은 평가 프로그램 과정에서 최종 사용자 라이센스 동의 후 별도의 채널을 통해 전달한다.

  * ``furiosa-libnux-[x.y.z]_amd64.deb``:  FuriosaAI NPU Runtime 라이브러리
  * ``furiosa-libnux-dev-[x.y.z]_amd64.deb``: FuriosaAI NPU Runtime 개발 환경 라이브러리 (C 헤더 파일, 정적 라이브러리 등)

``x.y.z`` 를 다운받은 SDK 버전에 맞게 변경하고 아래 명령으로 시스템에 설치하자.

.. code-block::

  $ apt-get install -y ./furiosa-libnux-[x.y.z]_amd64.deb
  $ apt-get install -y ./furiosa-libnux-dev-[x.y.z]-dev_amd64.deb
