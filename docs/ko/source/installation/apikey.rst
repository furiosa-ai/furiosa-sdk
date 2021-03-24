**********************************
API 키 설정
**********************************

FuriosaAI SDK의 웹서비스 기반의 도구를 사용하기 위해서는
웹서비스 API 키를 설정해야 합니다.

.. note::

  만약 API 키가 보유하고 있지 않다면 contact@furiosa.ai 으로 메일을 보내
  FuriosaAI SDK 및 NPU 평가판 사용에 대해 문의하시기 바랍니다.


소유한 API 키를 ``$HOME/.furiosa/credentials`` 파일에 아래와 쓰고 저장합니다.


.. code-block:: sh

  FURIOSA_ACCESS_KEY_ID=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  FURIOSA_SECRET_ACCESS_KEY=YYYYYYYYYYYYYYYYYYYYYYYYYYYYY


인증 확인
==========================
FuriosaAI SDK에 포함된 명령어 도구 또는 Python SDK를 사용하면 
API 키가 올바르게 설정되었는지 확인할 수 있습니다.

명령어 도구를 사용한 인증 확인
-----------------------------------------
.. code-block:: sh

  $ furiosa toolchain list    
  Available Toolchains:
  [0] 0.1.0 (rev: 952707e5f built_at: 2020-12-15 23:38:22)


만약 API 키가 잘못되었거나 설정이 올바르지 않을 경우 아래와 같은 오류를 볼 수 있습니다.

.. code-block:: sh

  $ furiosa toolchain list
  Client version: .dev0+bd0a54fdfba11243139668eff5fdf5dccfe4c470.dirty
  ERROR: fail to get version (http_status: 401, error_code: InvalidAPIKey, message: Your API Key is invalid. Please use a correct API key.
