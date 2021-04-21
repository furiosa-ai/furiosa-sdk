**********************************
API Key Settings
**********************************

The FuriosaAI NPU Runtime and FuriosaAI NPU Python SDK include web service based tools provided by FuriosaAI;
to use these tools, you need to set a web service API key.

.. note::

  If you don't have an API key, you can get one through the FuriosaAI SDK and NPU evaluation program process
  by sending a request to contact@furiosa.ai.

Write and save your own API key in ``$HOME/.furiosa/credentials`` file as follows.

.. code-block:: sh

  FURIOSA_ACCESS_KEY_ID=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  FURIOSA_SECRET_ACCESS_KEY=YYYYYYYYYYYYYYYYYYYYYYYYYYYYY


Authentication Testing
======================
You can verify that the API key is set up correctly by using the Python SDK
or the command line tools included with the FuriosaAI SDK.

Authentication Testing using Command Line Tools
-----------------------------------------------
.. code-block:: sh

  $ furiosa toolchain list    
  Available Toolchains:
  [0] 0.1.0 (rev: 952707e5f built_at: 2020-12-15 23:38:22)


If the API key is invalid or not set properly, you will see the following error.

.. code-block:: sh

  $ furiosa toolchain list
  Client version: .dev0+bd0a54fdfba11243139668eff5fdf5dccfe4c470.dirty
  ERROR: fail to get version (http_status: 401, error_code: InvalidAPIKey, message: Your API Key is invalid. Please use a correct API key.
