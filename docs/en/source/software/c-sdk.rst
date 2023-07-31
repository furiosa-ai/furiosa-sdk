.. _CSDK:

*********************************************************
C SDK installation and user guide
*********************************************************

We explain here how to write FuriosaAI NPU applications using C programming language.
The C SDK provides a C ABI-based static library and C header file. Using these, you can write applications in C, C++, or other languages that support C ABI.

The provided C SDK is relatively lower-level than :ref:`Python SDK <PythonSDK>`. It can be used when lower latency and higher performance are required, or when Python runtime cannot be used.

.. warning::

  ``furiosa-libnux-dev`` and the current C API are being deprecated in the future release.

  As substitute of the current API, new C API based on the next-generation runtime called FuriosaRT
  will be introduced with more features in the future release.


C SDK installation
===================================

The minimum requirements for C SDK are as follows.

* Ubuntu 20.04 LTS (Debian bullseye) or higher
* System administrator privileges (root)
* :ref:`FuriosaAI SDK required packages <RequiredPackages>`

In order to install and use C SDK, you must install the driver, firmware, and runtime library in accordance with
the :ref:`Required Package Installation <RequiredPackages>` guide.

Once you have installed the required packages, follow the instructions below to install C SDK.

.. tabs::

  .. tab:: Installation using APT server

    To use FuriosaAI APT, refer to :ref:`SetupAptRepository` and complete the authentication setting for server connection.

    .. code-block:: sh

      apt-get update && apt-get install -y furiosa-libnux-dev

  .. .. tab:: Installation using download center

  ..   Log in to the download center and download the latest versions of the packages below.

  ..   * NPU C SDK download (furiosa-libnux-dev-x.y.z-?.deb)

  ..   .. code-block:: sh

  ..     $ apt-get install -y ./furiosa-libnux-dev-x.y.z-?.deb


Compiling with C SDK
===================================
Once you install the package as above, you can compile using the C SDK.

C header files and static libraries are located in the ``/usr/include/furiosa``
and ``/usr/lib/x86_64-linux-gnu`` directories respectively.
They are the system paths that gcc looks to find C headers and libraries by default,
so you can simply compile C applications with only ``-lnux`` option as the following:


.. code-block:: sh

  gcc example.c -lnux


Also, you can find C SDK examples and C API reference at
`C Language SDK Reference <https://furiosa-ai.github.io/docs/v0.10.0/en/api/c/index.html>`_.
