****************************************************
FuriosaAI NPU & SDK |release| Documents
****************************************************

This document explains FuriosaAI NPU and its SDKs.

.. note::

   FuriosaAI software components include kernel driver, firmware, runtime, C SDK, Python SDK,
   and command lines tools. Currently, we offer them for only users who register *Early
   Access Program (EAP)* and agree to *End User Licence Agreement (EULA)*.
   Please contact contact@furiosa.ai to learn how to start the EAP.


FuriosaAI NPU
-------------------------------------------------
* :doc:`Introduction to FuriosaAI Warboy </npu/warboy>`: HW specification, performance, and supported operators

FuriosaAI Software
-------------------------------------------------
* :doc:`/software/intro`
* :doc:`/software/installation`
* :doc:`/software/python-sdk`
* :doc:`/software/c-sdk`
* :doc:`/software/cli`
* :doc:`/software/compiler`
* :doc:`/software/quantization`
* `FuriosaAI Model Zoo <https://furiosa-ai.github.io/furiosa-models/latest/>`_
* :doc:`/software/kubernetes_support`

FuriosaAI SDK Tutorial and Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `Tutorial: How to use Furiosa SDK from Start to Finish <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/HowToUseFuriosaSDKFromStartToFinish.ipynb>`_
* `Tutorial: Basic Inference API <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/GettingStartedWithPythonSDK.ipynb>`_
* `Tutorial: Advanced Inference API <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/AdvancedTopicsInInferenceAPIs.ipynb>`_
* `Example: Comparing Accuracy with CPU-based Inference <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/InferenceAccuracyCheck.ipynb>`_
* `Example: Image Classification Inference <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/Image_Classification.ipynb>`_
* `Example: SSD Object Detection Inference <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/SSD_Object_Detection.ipynb>`_
* `Other Python SDK Examples <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/inferences>`_

Serving, Model Deployment, MLOps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :doc:`/software/serving`
* :doc:`/software/kubernetes_support`

References
-------------------------------------------------
* `C SDK Reference <https://furiosa-ai.github.io/docs/v0.5.0/en/api/c/index.html>`_
* :doc:`Python API Reference </api/python/modules>`

Other Links
--------------------------------------------------
* `FuriosaAI Home <https://furiosa.ai>`_
* `FuriosaAI Customer Support Center <https://furiosa-ai.atlassian.net/servicedesk/customer/portals/>`_
* :ref:`BugReport`


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: FuriosaAI NPU

   /npu/warboy

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: FuriosaAI Software

   /software/intro
   /software/installation
   /software/python-sdk
   /software/c-sdk
   /software/cli
   /software/compiler
   /software/quantization
   /software/profiler
   FuriosaAI Model Zoo <https://furiosa-ai.github.io/furiosa-models/latest/>
   /software/serving
   /software/kubernetes_support
   /software/tutorials
   /software/references


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Release Notes

   /releases/0.9.0.rst
   /releases/0.8.0.rst
   /releases/0.7.0.rst
   /releases/0.6.0.rst
   /releases/0.5.0.rst


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Customer Support

   FuriosaAI Customer Center <https://furiosa-ai.atlassian.net/servicedesk/customer/portals/>
   /customer-support/bugs

..
   Download Center <https://developer.furiosa.ai/downloads/>


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Previous Documents

   v0.8.0 <https://furiosa-ai.github.io/docs/v0.8.0/en/>
   v0.6.0 <https://furiosa-ai.github.io/docs/v0.6.0/en/>
   v0.5.0 <https://furiosa-ai.github.io/docs/v0.5.0/en/>
   v0.2.0 <https://furiosa-ai.github.io/docs/v0.2.0/en/>
