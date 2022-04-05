.. _CSDK:

*********************************************************
Installation and User Guide of C SDK
*********************************************************

Using the Furiosa C SDK library, you can easily write C and C++ application programs that leverage the NPU.
C SDK provides C ABI based static library and C header files for not only C and C++ applications but also other program applications which support C ABI.

C SDK provides low-level API compared to :ref:`Python SDK <PythonSDK>`, which enables lower latency and higher performance than using Python runtime.
C SDK also provides blocking and asynchronous API as Python SDK provides.

Requirements for installation
===================================

* Ubuntu 18.04 LTS (Debian buster) or higher version
* System root privileges
* :ref:`FuriosaAI SDK Required Packages <RequiredPackages>`

In order to use C SDK, kernel driver, firmware, and runtime library must be installed (please refer to :ref:`FuriosaAI SDK Required Packages <RequiredPackages>`)

.. tabs::

  .. tab:: Installation with APT server

    In order to use FuriosaAI API, authentication for accessing Furiosa server is required. Please refer to :ref:`SetupAptRepository` details.

    .. code-block:: sh

      apt-get update && apt-get install -y furiosa-libnux-dev

  .. tab:: Installation with download center

    Please login to Furiosa download center for downloading the latest versions of the following packages.

    * NPU C SDK (furiosa-libnux-dev-x.y.z-?.deb)

    .. code-block:: sh

      $ apt-get install -y ./furiosa-libnux-dev-x.y.z-?.deb


Compilation with C SDK
===================================

.. warning::
  TODO

