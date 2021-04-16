**********************************
NPU Runtime Installation
**********************************

FuriosaAI NPU Runtime includes the NPU runtime for accelerating a deep learning model on the NPU,
an interface for executing it, and various tools required for NPU application development.
FuriosaAI NPU Runtime can be used by installing the following two packages, while the deb package 
configuration is delivered through a separate channel after the accepting the End User License Agreement 
(under the evaluation program process).

  * ``furiosa-libnux-[x.y.z]_amd64.deb``:  FuriosaAI NPU Runtime library
  * ``furiosa-libnux-dev-[x.y.z]_amd64.deb``: FuriosaAI NPU Runtime  development environment library (C header file, static library, etc)

Change ``x.y.z`` according to your downloaded SDK version and install it on the system with the below. 

.. code-block::

  $ apt-get install -y ./furiosa-libnux-[x.y.z]_amd64.deb
  $ apt-get install -y ./furiosa-libnux-dev-[x.y.z]-dev_amd64.deb
