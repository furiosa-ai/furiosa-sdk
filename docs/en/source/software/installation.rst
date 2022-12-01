.. _RequiredPackages:

************************************************************
Driver, Firmware, and Runtime Installation
************************************************************

Here, we explain how to install the packages necessary to use
the various SW components provided by FuriosaAI. 

These required packages are kernel drivers, firmware, and runtime library,
and can be downloaded either directly through the download center,
or through the APT/PIP servers as issued on the developer site.

.. note::

  The download center and developer site will be provided upon registration
  to the FuriosaAI evaluation program. Currently, the request for registration
  can be done through contact@furiosa.ai.

.. _MinimumRequirements:

Minimum requirements for SDK installation
=====================================================================
* Ubuntu 18.04 LTS (Bionic Beaver)/Debian buster
  or higher
* Administrator privileges on system (root)
* Internet-accessible network


.. _SetupAptRepository:

APT server configuration
=====================================================================

In order to use the APT server as provided by FuriosaAI, the APT server must be configured
on Ubuntu or Debian Linux as delineated below.
This section may be skipped if you are using the download center, and not the APT.


1. Install the necessary packages to access HTTPS-based APT server.

.. code-block:: sh

  sudo apt update
  sudo apt install -y ca-certificates apt-transport-https gnupg

2. Register the FuriosaAI public Singing key.

.. code-block:: sh

  sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key 5F03AFA423A751913F249259814F888B20B09A7E

3. Issue the API key from FuriosaAI developer center, and configure the API key as follows:


.. code-block:: sh

  sudo tee -a /etc/apt/auth.conf.d/furiosa.conf > /dev/null <<EOT
    machine archive.furiosa.ai
    login xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    password xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  EOT

  sudo chmod 400 /etc/apt/auth.conf.d/furiosa.conf


4. Configure the APT server according to the explanation given in the Linux distribution version tab.


.. tabs::

  .. tab:: Ubuntu 18.04 (Debian Buster)

      Register the APT server through the command below.

      .. code-block:: sh

        sudo tee -a /etc/apt/sources.list.d/furiosa.list <<EOT
        deb [arch=amd64] https://archive.furiosa.ai/ubuntu bionic restricted
        EOT

  .. tab:: Ubuntu 20.04 (Debian Bullseye)

      Register the APT server through the command below.

      .. code-block:: sh

        sudo tee -a /etc/apt/sources.list.d/furiosa.list <<EOT
        deb [arch=amd64] https://archive.furiosa.ai/ubuntu focal restricted
        EOT



.. _InstallLinuxPackages:

Installing required packages.
=====================================================================

If you have registered the APT server as above, or registered on the download site,
you will be able to install the required packages - NPU kernel driver, firmware, and runtime.

.. tabs::

  .. tab:: Installation using APT server

    .. code-block:: sh

      sudo apt-get update && sudo apt-get install -y \
      furiosa-driver-pdma furiosa-libnpu-warboy furiosa-libnux libonnxruntime

  .. tab:: Installation using download center

    Select the latest version of the packages below, download them,
    and install them in order as written in the command.

    * NPU Driver (furiosa-driver-pdma)
    * Firmware (furiosa-libnpu)
    * Runtime library  (furiosa-libnux)
    * Onnxruntime  (libonnxruntime)

    .. code-block:: sh

      sudo apt-get install -y ./furiosa-driver-pdma-x.y.z-?.deb
      sudo apt-get install -y ./furiosa-libnpu-warboy-x.y.z-?.deb
      sudo apt-get install -y ./libonnxruntime-x.y.z-?.deb
      sudo apt-get install -y ./furiosa-libnux-x.y.z-?.deb


Holding/unholding installed version
------------------------------------

Following package installation, in order to maintain a stable operating environment,
there may be a need to hold the installed packages versions. By using the command below,
you will be able to hold the currently installed versions.

.. code-block:: sh

  sudo apt-mark hold furiosa-driver-pdma furiosa-libnpu-warboy furiosa-libnux libonnxruntime


In order to unhold and update the current package versions, designate the package
that you wish to unhold with the command ``apt-mark unhold``.
Here, you can state the name of the package, thereby unholding selectively
a specific package. In order to show the properties of an already held package,
use the command ``apt-mark showhold``.

.. code-block:: sh

  sudo apt-mark unhold furiosa-driver-pdma furiosa-libnpu-warboy furiosa-libnux libonnxruntime


Installing a specific version
------------------------------

If you need to install a specific version,
you may designate the version that you want and install as follows.

1. Check available versions through ``apt list``.

.. code-block:: sh

  sudo apt list -a furiosa-libnux


2. State the package name and version as options in the command ``apt-get install``

.. code-block:: sh

  sudo apt-get install -y furiosa-libnux=0.6.0-2
