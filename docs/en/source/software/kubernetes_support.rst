.. _KubernetesIntegration:

**********************************
Kubernetes Support
**********************************

`Kuberentes <https://kubernetes.io/>`_ is an open source platform for managing containerized workloads and services.
Furiosa SDK provides the following components to support the Kubernetes environment.

* FuriosaAI NPU Device Plugin (`Introduction to Kubernetes Device Plugin <https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/>`_)

* FuriosaAI NPU Feature Discovery (`Introduction to Node Feature Discovery <https://kubernetes-sigs.github.io/node-feature-discovery/stable/get-started/index.html>`_)

The two components above provide the following functions.

* Make the Kubernetes cluster aware of the NPUs available to the node.
* Through Kubernetes ``spec.containers[].resources.limits`` , schedule the NPU simultaneously when distributing Pod workload.
* Identify NPU information of NPU-equipped machine, and register it as node label (you can selectively schedule Pods with this information and `nodeSelector`)
   * The node-feature-discovery needs to be installed to the cluster, and the ``nfd-worker`` Pod must be running in the nodes equipped with NPUs.

The setup process for Kubernetes support is as follows.

1. Preparing NPU nodes
========================================
Requirements for Kubernetes nodes are as follows.

* Ubuntu 18.04, 20.04 or higher
* Intel compatible CPU

You also need to install NPU driver and toolkit on each node of NPU-equipped Kubernetes.
If the APT server is set up (see :ref:`SetupAptRepository`), you can easily install as follows.

.. code-block:: sh

  apt-get update && apt install -y furiosa-driver-pdma furiosa-toolkit


Once the required package is installed as above, you can check for NPU recognition as follows, with the
``furiosactl`` command included in furiosa-toolkit.
If the NPU is not recognized with the command below, try again after rebooting - depending on the environment.

.. code-block:: sh

  $ furiosactl info
  +------+------------------+-------+--------+--------------+---------+
  | NPU  | Name             | Temp. | Power  | PCI-BDF      | PCI-DEV |
  +------+------------------+-------+--------+--------------+---------+
  | npu0 | FuriosaAI Warboy |  40°C | 1.37 W | 0000:01:00.0 | 509:0   |
  +------+------------------+-------+--------+--------------+---------+

2. Installing Node Feature Discovery
=========================================
In order to make Kubernetes to recognize NPUs, you need to install Node Feature Discovery.
By running the command as shown in the example below, if there is a node label that begins with ``feature.node.kubernetes.io/...``, Node Feature Discovery's DaemonSet has already been installed

.. code-block:: sh

  $ kubectl get no -o json | jq '.items[].metadata.labels'
  {
    "beta.kubernetes.io/arch": "amd64",
    "beta.kubernetes.io/os": "linux",
    "feature.node.kubernetes.io/cpu-cpuid.ADX": "true",
    "feature.node.kubernetes.io/cpu-cpuid.AESNI": "true",
    ...

* If you do not have the Node Feature Discovery in your cluster, refer to the following document.

   * `Quick start / Installation <https://kubernetes-sigs.github.io/node-feature-discovery/v0.11/get-started/quick-start.html#installation>`_

* The following options must be applied when executing Node Feature Discovery.

  * ``beta.furiosa.ai`` needs to be included in the ``--extra-label-ns`` option of ``nfd-master``

  * In the config file of ``nfd-worker``,
    * Only ``vendor`` in the ``sources.pci.deviceLabelFields`` value
    *  ``"12"`` must be included as a value in ``sources.pci.deviceClassWhitelist``


.. note::

  Installing Node Feature Discovery is not mandatory, but is recommended. The next step
will explain the additional tasks that must be performed if you are not using
Node Feature Discovery.


.. _InstallingDevicePluginAndNfd:

3. Installing Device Plugin and NPU Feature Discovery
==========================================================

When the NPU node is ready, install Device Plugin and NPU Feature Discovery's DaemonSet as follows.

.. code-block:: sh

    kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/v0.7.0/kubernetes/deployments/device-plugin.yaml
    kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/v0.7.0/kubernetes/deployments/npu-feature-discovery.yaml

After executing the above command, you can check whether the installed daemonset is functioning normally with the ``kubectl get daemonset -n kube-system`` command.
For reference, the DaemonSet is distributed only to nodes equipped with NPUs, and uses
``alpha.furiosa.ai/npu.family=warboy`` information that the Node Feature Discovery (``feature.node.kubernetes.io/pci-1ed2.present=true``) attaches to each node.

.. code-block:: sh

  $ kubectl get daemonset -n kube-system
 NAME                           DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                                      AGE
  furiosa-device-plugin          3         3         3       3            3           feature.node.kubernetes.io/pci-1ed2.present=true   128m
  furiosa-npu-feature-discovery  3         3         3       3            3           feature.node.kubernetes.io/pci-1ed2.present=true   162m

The metadata attached by the Node Feature Discovery is shown in the following table.

.. _K8sNodeLabels:

.. list-table:: NPU Node Labels
   :widths: 50 50 50
   :header-rows: 1

   * - Label
     - Value
     - Description
   * - beta.furiosa.ai/npu.count
     - 1
     - The number of NPUs e x b number of NPUs attached to node
   * - beta.furiosa.ai/npu.product
     - warboy, warboyB0
     - NPU Product Name (Code)
   * - beta.furiosa.ai/npu.family
     - warboy, renegade
     - NPU Architecture (Family)
   * - beta.furiosa.ai/machine.vendor
     - (depends on machine)
     - Machine Manufacturer
   * - beta.furiosa.ai/machine.name
     - (depends on machine)
     - The Nmae of Machine (Code)
   * - beta.furiosa.ai/driver.version
     - 1.3.0
     - NPU Device Driver Version
   * - beta.furiosa.ai/driver.version.major
     - 1
     - Major Version Number of NPU Device Driver Version
   * - beta.furiosa.ai/driver.version.minor
     - 3
     - Minor Version Number of NPU Device Driver
   * - beta.furiosa.ai/driver.version.patch
     - 0
     - Patch Version Number of NPU Device Driver
   * - beta.furiosa.ai/driver.reference
     - 57ac7b0
     - Build Commit Hash of NPU Device Driver

If you want to check node labels, then execute the ``kubectl get nodes --show-labels`` command. If you see labels which start with ``beta.furiosa.ai`` Node Feature Discovery is successfully installed.

.. code-block:: sh

    kubectl get nodes --show-labels

    warboy-node01     Ready   <none>  65d   v1.20.10   beta.furiosa.ai/npu.count=1,beta.furiosa.ai/npu.product=warboy...,kubernetes.io/os=linux
    warboy-node02     Ready   <none>  12d   v1.20.10   beta.furiosa.ai/npu.count=1,beta.furiosa.ai/npu.product=warboy...,kubernetes.io/os=linux


Device Plugin Configuration
--------------------------------------
Execution options for Device Plugin can be set by the argument of command line or configuration file.

1. Command Line Arguments

The option can be set by the ``k8s-device-plugin`` command as follows.

.. code-block:: sh

  $ k8s-device-plugin --interval 10

For the Pod or DaemonSet specification command line arguments can be set as follows.

.. code-block:: yaml

  apiVersion: v1
  kind: Pod
  metadata:
    name: furiosa-device-plugin
    namespace: kube-system
  spec:
    containers:
      - name: device-plugin
        image: ghcr.io/furiosa-ai/k8s-device-plugin:latest
        command: ["/usr/bin/k8s-device-plugin"]
        args: ["--interval", "10"]
  # (the reset is omitted)

.. list-table:: arguments of k8s-device-plugin
   :widths: 50 150 50
   :header-rows: 1

   * - Item
     - Explanation
     - Default Value
   * - default-pe
     - default core type when pod is allocated (Fusion/Single)
     - Fusion
   * - interval
     - interval for searching device (seconds)
     - 10
   * - disabled-devices
     - devices not for allocations (several devices can be designated using comma)
     -
   * - plugin-dir
     - directory path of kubelet device-plugin
     - /var/lib/kubelet/device-plugins
   * - socket-name
     - file name of socket created under <plugin-dir>
     - furiosa-npu
   * - resource-name
     - name of NPU resource registered for k8s node
     - beta.furiosa.ai/npu

2. Setting Configuration File

You may set configuration file by executing ``k8s-device-plugin`` command with argument ``config-file``.
If ``config-file`` is set then the other arguments are not permitted.

.. code-block:: sh

  $ k8s-device-plugin --config-file /etc/furiosa/device-plugin.conf

.. code-block:: yaml
   :caption: /etc/furiosa/device-plugin.conf

   interval: 10
   defaultPe: Fusion
   disabledDevices:             # device npu1 equipped in warboy-node01 will not be used
     - devName: npu1
       nodeName: warboy-node01
   pluginDir: /var/lib/kubelet/device-plugins
   socketName: furiosa-npu
   resourceName: beta.furiosa.ai/npu

Configuration file is a text file with Yaml format. The modification of file contents is applied to Device Plugin immediately. Updated configuration is recorded on log of Device Plugin.
(but, modifications on ``pluginDir`` , ``socketName``, or ``resourceName`` require reboot.)

:ref:`InstallingDevicePluginAndNfd` provides ``device-plugin.yaml`` which is default configuration file based on ConfigMap.
If you want to modify execution options of Device Plugin, modify ConfigMap. Once modified ConfigMap is applied to Pod, Device Plugin reads the ConfigMap and then reflects modification.

.. code-block:: sh

  $ kubectl edit configmap npu-device-plugin -n kube-system

.. code-block:: yaml
   :caption: configmap/npu-device-plugin

   apiVersion: v1
   data:
     config.yaml: |
       defaultPe: Fusion
       interval: 15
       disabledDevices:
         - devName: npu2
           nodeName: npu-001
   kind: ConfigMap

4. Creating a Pod with NPUs
====================================

To allocate NPU to a Pod, add as shown below to ``spec.containers[].resources.limits``.

.. code-block:: yaml

    resources:
        limits:
            beta.furiosa.ai/npu: "1" # requesting 1 NPU

`Full example <https://github.com/furiosa-ai/furiosa-sdk/blob/v0.7.0/kubernetes/deployments/pod-example.yaml>`_ for Pod creation is as follows.

.. code-block:: sh

  $ cat > npu-pod.yaml <<EOL
  apiVersion: v1
  kind: Pod
  metadata:
    name: npu-pod
  spec:
    containers:
      - name: npu-pod
        image: ubuntu:focal
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
            beta.furiosa.ai/npu: "1"
          requests:
            cpu: "4"
            memory: "8Gi"
            beta.furiosa.ai/npu: "1"
  EOL

  $ kubectl apply -f npu-pod.yaml

After Pod creation, you can check NPU allocation as follows.

.. code-block:: sh

  $ kubectl get pods npu-pod -o yaml | grep alpha.furiosa.ai/npu
      beta.furiosa.ai/npu: "1"
      beta.furiosa.ai/npu: "1"


The SDK application automatically recognizes the allocated NPU device.
If there are multiple NPU devices on a node, you can check which device is allocated as follows:

.. code-block:: sh

    $ kubectl exec npu-pod -it -- /bin/bash
    root@npu-pod:/# echo $NPU_DEVNAME
    npu0pe0-1


If furiosa-toolkit is installed in the Pod, you can check for more detailed device information using the
furiosactl command as shown below.

See :ref:`SetupAptRepository` for installation guide using APT.

.. code-block:: sh

    root@npu-pod:/# furiosactl
    furiosactl controls the FURIOSA NPU.

    Find more information at: https://furiosa.ai/

    Basic Commands:
        version    Print the furiosactl version information
        info       Show information one or many NPU(s)
        config     Get/Set configuration for NPU environment

    Usage:
        furiosactl COMMAND

    root@npu-pod:/# furiosactl info
    +------+------------------+-------+--------+--------------+---------+
    | NPU  | Name             | Temp. | Power  | PCI-BDF      | PCI-DEV |
    +------+------------------+-------+--------+--------------+---------+
    | npu0 | FuriosaAI Warboy |  40°C | 1.37 W | 0000:01:00.0 | 509:0   |
    +------+------------------+-------+--------+--------------+---------+

5. NPU monitoring
====================================

If you install ``npu-metrics-exporter``, its daemon set and service will be created in your kubernetes cluster.
The Pod that is executed through DaemonSet outputs various NPU status information that may be
useful for monitoring. The data is expressed in Prometheus format. If Prometheus
is installed, and service discovery is active, Prometheus will automatically collect
data through the Exporter.

The collected data may be reviewed with visualization tools such as Grafana.


.. list-table:: npu-metrics-exporter collection category list
   :widths: 250 250
   :header-rows: 1

   * - Name
     - Details
   * - furiosa_npu_alive
     - NPU operation status (1:normal)
   * - furiosa_npu_uptime
     - NPU operation time (s)
   * - furiosa_npu_error
     - Number of detected NPU errors
   * - furiosa_npu_hw_temperature
     - Temperature of each NPU components (°mC)
   * - furiosa_npu_hw_power
     - NPU instantaneous power usage (µW)
   * - furiosa_npu_hw_voltage
     - NPU instantaenous voltage (mV)
   * - furiosa_npu_hw_current
     - NPU instantaneous current (mA)