.. _KubernetesIntegration:

**********************************
Kubernetes Support
**********************************

`Kuberentes <https://kubernetes.io/>`_ is an open source platform for managing containerized workloads and services.
Furiosa SDK provides the following components to support the Kubernetes environment.

* `Kubernetes Device Plugin <https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/>`_
* Kubernetes Node Labeller

The two components above provide the following functions.

* Make the Kubernetes cluster aware of the NPUs available to the node.
* Through Kubernetes ``spec.containers[].resources.limits`` , schedule the NPU simultaneously when distributing Pod workload.
* Identify NPU information of NPU-equipped machine, and register it as node label (you can selectively schedule Pods with this information and `nodeSelector`)

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
furiosactl command included in furiosa-toolkit.
If the NPU is not recognized with the command below, try again after rebooting - depending on the environment.

.. code-block:: sh

  $ furiosactl info
  +------+------------------+-------+--------+--------------+---------+
  | NPU  | Name             | Temp. | Power  | PCI-BDF      | PCI-DEV |
  +------+------------------+-------+--------+--------------+---------+
  | npu1 | FuriosaAI Warboy |  40°C | 1.37 W | 0000:01:00.0 | 509:0   |
  +------+------------------+-------+--------+--------------+---------+

2. Installing Device Plugin, Node Labeller
==============================================

Once NPU node preparation is complete, install the device plugin and node labeller (daemonset) as follows.

.. code-block:: sh

  kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/v0.6.0/kubernetes/deployments/node-labeller.yaml
  kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/v0.6.0/kubernetes/deployments/device-plugin.yaml

After executing the above command, you can check whether the installed daemonset is functioning normally with the ``kubectl get daemonset -n kube-system`` command.
For reference, the device plugin (``furiosa-npu-plugin``) is only distributed to nodes equipped with NPUs, and uses
``alpha.furiosa.ai/npu.family=warboy`` information that the node labeller (``furiosa-npu-labeller``) attaches to each node.

.. code-block:: sh

  $ kubectl get daemonset -n kube-system
  NAME                     DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                                               AGE
  furiosa-npu-labeller     6         6         6       6            6           kubernetes.io/os=linux                                      321d
  furiosa-npu-plugin       2         2         2       2            2           alpha.furiosa.ai/npu.family=warboy,kubernetes.io/os=linux   159d

The metadata attached by the node labeller (``furiosa-npu-labeller``) is shown in the following table.

.. _K8sNodeLabels:

.. list-table:: NPU Node Labels
   :widths: 50 50 50
   :header-rows: 1

   * - Label
     - Value
     - Description
   * - alpha.furiosa.ai/npu.family
     - warboy, renegade
     - Chip family
   * - alpha.furiosa.ai/npu.hwtype
     - haps (ASIC), u250 (FPGA sample)
     - HW type


If you execute the ``kubectl get nodes --show-labels`` command to check node labels, and you see labels starting with ``alpha.furiosa.ai``
as follows, you have successfully installed the node labeller.

.. code-block:: sh

  kubectl get nodes --show-labels

  warboy-node01     Ready   <none>  65d   v1.20.10   alpha.furiosa.ai/npu.family=warboy,alpha.furiosa.ai/npu.hwtype=haps...,kubernetes.io/os=linux
  warboy-node02     Ready   <none>  12d   v1.20.10   alpha.furiosa.ai/npu.family=warboy,alpha.furiosa.ai/npu.hwtype=haps...,kubernetes.io/os=linux


3. Creating a Pod with NPUs
====================================

To allocate NPU to a Pod, add as shown below to ``spec.containers[].resources.limits``.

.. code-block:: yaml

  resources:
    limits:
      alpha.furiosa.ai/npu: "1" # requesting 1 NPU


`Full example <https://github.com/furiosa-ai/furiosa-sdk/blob/v0.6.0/kubernetes/deployments/pod-example.yaml>`_ for Pod creation is as follows.

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
            alpha.furiosa.ai/npu: "1"
          requests:
            cpu: "4"
            memory: "8Gi"
            alpha.furiosa.ai/npu: "1"
  EOL

  $ kubectl apply -f npu-pod.yaml

After Pod creation, you can check NPU allocation as follows.

.. code-block:: sh

  $ kubectl get pods npu-pod -o yaml | grep alpha.furiosa.ai/npu
      alpha.furiosa.ai/npu: "1"
      alpha.furiosa.ai/npu: "1"


If there are multiple NPU devices, you can check which devices are allocated as follows.
The SDK application automatically recognizes the allocated NPU device.

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
  | npu1 | FuriosaAI Warboy |  40°C | 1.37 W | 0000:01:00.0 | 509:0   |
  +------+------------------+-------+--------+--------------+---------+