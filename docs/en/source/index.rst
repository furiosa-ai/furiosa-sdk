.. Furiosa SDK Documents master file, created by
   sphinx-quickstart on Tue Mar 23 11:18:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FuriosaAI SDK Documentation
=================================================

This document covers the process for installing and using the FuriosaAI SDK.

.. note::

   The FuriosaAI SDK includes command line tools, runtime libraries, and Python libraries.
   FuriosaAI NPU's kernel driver, firmware, and runtime are distributed according to
   FuriosaAI's evaluation program registration and End User License Agreement.
   For questions about downloading and evaluating the program, please contact us at contact@furiosa.ai.

SDK Installation
-------------------------------------------------
* :doc:`SDK Installation Prerequisites (Required) </installation/prerequisites>`
* :doc:`NPU Kernel Driver and Firmware Installation (Required) </installation/driver>`
* :doc:`NPU Runtime Installation (Required)</installation/runtime>` : NPU Runtime and Application Tools
* :doc:`NPU Python SDK Installation (Optional)</installation/python-sdk>` : Python Library and Command Line Tool for NPU Usage
* :doc:`Web Service API key setting (Optional)</installation/apikey>` : API Key Setting for Using Tools Provided as Web Services

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: SDK installation

   /installation/prerequisites
   /installation/driver
   /installation/runtime
   /installation/python-sdk
   /installation/apikey

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Quickstart

   /quickstart/python-sdk
   /quickstart/cli

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Advanced Topics

   /advanced/quantization
   /advanced/supported_operators

Quickstart
-------------------
* :doc:`Command Line Tools Quickstart </quickstart/cli>`
* :doc:`Python SDK Quickstart </quickstart/python-sdk>`

Advanced Topics
-------------------
* :doc:`Model Quantization </advanced/quantization>`
* :doc:`NPU Acceleration Operators List </advanced/supported_operators>`


Tutorials
-----------------------------------
* `Getting Started with Python SDK <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/GettingStartedWithPythonSDK.ipynb>`_
* `Advanced Inference API <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/AdvancedTopicsInInferenceAPIs.ipynb>`_
* `How to Use Furiosa SDK from Start to Finish <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/HowToUseFuriosaSDKFromStartToFinish.ipynb>`_

Code Examples
------------------------------------
* `Basic Inference <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/inferences>`_
* `Quantization <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/quantizers>`_
* `Comparing the accuracy between NPU and CPU inferences <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/InferenceAccuracyCheck.ipynb>`_
* `Image Classification Inference Example <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/Image_Classification.ipynb>`_
* `SSD Object Detection Inference Example <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/SSD_Object_Detection.ipynb>`_

References
-------------------
* `C API Reference <https://furiosa-ai.github.io/docs/v0.5.0/en/api/c/index.html>`_
* :doc:`Python API Reference </api/python/modules>`

Other links
-------------------
* `FuriosaAI Homepage <https://furiosa.ai>`_
* `FuriosaAI SDK Github <https://github.com/furiosa-ai/furiosa-sdk>`_

