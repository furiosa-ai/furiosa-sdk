.. _RequiredPackages:

************************************************************
Driver, Firmware, and Runtime Installation
************************************************************

Here, we explain how to install the packages necessary to use
the various SW components provided by FuriosaAI.
The required packages are composed of kernel drivers, firmware, and runtime library,
and they can be easily installed via the APT package manager.

.. note::

  You will be able to login `FuriosaAI IAM <https://iam.furiosa.ai>`_ and create a new API key
  upon registration to the FuriosaAI evaluation program.
  Currently, the request for registration can be done through contact@furiosa.ai.

.. _MinimumRequirements:

Minimum requirements for SDK installation
=====================================================================
* Ubuntu 20.04 LTS (Focal Fossa)/Debian bullseye
  or higher
* Administrator privileges on system (root)
* Internet-accessible network


.. _SetupAptRepository:

APT server configuration
=====================================================================

In order to use the APT server as provided by FuriosaAI, the APT server must be configured
on Ubuntu or Debian Linux as delineated below.


1. Install the necessary packages to access HTTPS-based APT server.

.. code-block:: sh

  sudo apt update
  sudo apt install -y ca-certificates apt-transport-https gnupg wget

2. Register the FuriosaAI public Signing key.

.. code-block:: sh

  mkdir -p /etc/apt/keyrings && \
  wget -q -O- https://archive.furiosa.ai/furiosa-apt-key.gpg \
  | gpg --dearmor \
  | sudo tee /etc/apt/keyrings/furiosa-apt-key.gpg > /dev/null

3. Generate a new API key from `FuriosaAI IAM <https://iam.furiosa.ai>`_, and configure the API key as follows:


.. code-block:: sh

  sudo tee -a /etc/apt/auth.conf.d/furiosa.conf > /dev/null <<EOT
    machine archive.furiosa.ai
    login [KEY (ID)]
    password [PASSWORD]
  EOT

  sudo chmod 400 /etc/apt/auth.conf.d/furiosa.conf


4. Configure the APT server according to the explanation given in the Linux distribution version tab.


.. tabs::

  .. tab:: Ubuntu 20.04 (Debian Bullseye)

      Register the APT server through the command below.

      .. code-block:: sh

        sudo tee -a /etc/apt/sources.list.d/furiosa.list <<EOT
        deb [arch=amd64 signed-by=/etc/apt/keyrings/furiosa-apt-key.gpg] https://archive.furiosa.ai/ubuntu focal restricted
        EOT

  .. tab:: Ubuntu 22.04 (Debian Bookworm)

      Register the APT server through the command below.

      .. code-block:: sh

        sudo tee -a /etc/apt/sources.list.d/furiosa.list <<EOT
        deb [arch=amd64 signed-by=/etc/apt/keyrings/furiosa-apt-key.gpg] https://archive.furiosa.ai/ubuntu jammy restricted
        EOT



.. _InstallLinuxPackages:

Installing required packages.
=====================================================================

If you have registered the APT server as above, or registered on the download site,
you will be able to install the required packages - NPU kernel driver, firmware, and runtime.

.. tabs::

  .. tab:: Installation using APT server

    .. code-block:: sh

      sudo apt-get update && sudo apt-get install -y furiosa-driver-warboy furiosa-libnux

  .. .. tab:: Installation using download center

  ..   Select the latest version of the packages below, download them,
  ..   and install them in order as written in the command.
  ..   Update the ``x.y.z-?`` version portions in accordance with the downloaded files.


  ..   * NPU Driver (furiosa-driver-warboy)
  ..   * Firmware (furiosa-libhal)
  ..   * Runtime library  (furiosa-libnux)
  ..   * Onnxruntime  (libonnxruntime)

  ..   .. code-block:: sh

  ..     sudo apt-get install -y ./furiosa-driver-warboy-x.y.z-?.deb
  ..     sudo apt-get install -y ./furiosa-libhal-warboy-x.y.z-?.deb
  ..     sudo apt-get install -y ./libonnxruntime-x.y.z-?.deb
  ..     sudo apt-get install -y ./furiosa-libnux-x.y.z-?.deb


.. _AddUserToFuriosaGroup:

Adding a user to the ``furiosa`` Group
-----------------------------------------

Linux is a multi-user operating system that enables file and device access for both the owner and users within a specific group.
The NPU device driver creates a group called ``furiosa`` and restricts access to NPU devices exclusively to users who are members of the ``furiosa`` group.
To add a user to a member of ``furiosa`` group, please run as follows:

.. code-block:: sh

  sudo usermod -aG furiosa <username>


Replace <username> with the name of the user you want to add to the ``furiosa`` group.
For example, in order to add the current user (i.e., ``$USER``) to the ``furiosa`` group, you can run as follows:

.. code-block:: sh

  sudo usermod -aG furiosa $USER


Upon logging out and logging back in, the change to the group membership will take effect.


.. _HoldingAptVersion:

Holding/unholding installed version
------------------------------------

Following package installation, in order to maintain a stable operating environment,
there may be a need to hold the installed packages versions. By using the command below,
you will be able to hold the currently installed versions.

.. code-block:: sh

  sudo apt-mark hold furiosa-driver-warboy furiosa-libhal-warboy furiosa-libnux libonnxruntime


In order to unhold and update the current package versions, designate the package
that you wish to unhold with the command ``apt-mark unhold``.
Here, you can state the name of the package, thereby unholding selectively
a specific package. In order to show the properties of an already held package,
use the command ``apt-mark showhold``.

.. code-block:: sh

  sudo apt-mark unhold furiosa-driver-warboy furiosa-libhal-warboy furiosa-libnux libonnxruntime


.. _InstallSpecificVersion:

Installing a specific version
------------------------------

If you need to install a specific version,
you may designate the version that you want and install as follows.

1. Check available versions through ``apt list``.

.. code-block:: sh

  sudo apt list -a furiosa-libnux


2. State the package name and version as options in the command ``apt-get install``

.. code-block:: sh

  sudo apt-get install -y furiosa-libnux=0.9.1-?


.. _UpgradeFirmware:

NPU Firmware Update
=====================================================================

