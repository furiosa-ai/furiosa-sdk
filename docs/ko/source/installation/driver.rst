**********************************
커널 드라이버 및 펌웨어 설치
**********************************

``furiosa-fpga-install`` 는 FuriosaAI의 커널 드라이버와 펌웨어를 자동으로 설치하는 프로그램이다.
furiosa-fpga-install 스크립트를 이용하여 아래와 같이 설치한다.

보유한 하드웨어가 Alveo U250 이면 :ref:`AlveoU250` 를 따라 커널 드라이버와 펌웨어를 설치하고,
AWS F1 환경이라면 :ref:`AWS F1` 를 따라 설치한다.

.. _AlveoU250:

Alveo U250 가속기를 위한 설치
**********************************

.. code-block::

  $ git clone https://github.com/furiosa-ai/furiosa-fpga-install.git
  $ cd furiosa-fpga-install
  $ sudo ./install_furiosa_fpga_u250

  Furiosa AI's F1 SDK has been successfully installed. 
  Please REBOOT this machine to complete the installation.


함께 포함된 ``check_fpga_device`` 명령어로
설치가 성공적으로 되었는지 확인할 수 있다.


.. code-block::

  $ cd furiosa-fpga-install
  $ ./check_fpga_device
  [OK] Furiosa AI's FPGA device is detected.


.. _AWS F1:

AWS F1 환경에서 설치
*********************************

요구 사항
---------------------------------
* FuriosaAI의 AWS FPGA 이미지 접근 권한이 필요 하다. AWS account ID를 Furiosa AI에게 공유하면 권한을 받을 수 있다.
* AWS F1 인스턴스가 필요하며 현재 f1.2xlarge 타입만 지원 된다.

AWS Account ID 찾는 방법
---------------------------------------
AWS account에 로그인 한 뒤에 우측 최상단에 Account 이름을 클릭하면 
팝업 창이 뜬다. My account 옆에 있는 빨간 박스에 위치한 12 자리 숫자가 AWS Account ID 이다.

.. image:: ../../../imgs/aws_account_id.png

설치
===================
.. code-block::

  $ git clone https://github.com/furiosa-ai/furiosa-fpga-install.git
  $ cd furiosa-fpga-install
  $ sudo ./install_furiosa_fpga_u250

  ...
  Furiosa AI's F1 SDK has been successfully installed. 
  Please REBOOT this machine to complete the installation.

함께 포함된 ``check_fpga_device`` 명령어 도구로
설치가 성공적으로 되었는지 확인할 수 있다.


.. code-block::

  $ cd furiosa-fpga-install
  $ ./check_fpga_device
  [OK] Furiosa AI's FPGA device is detected.