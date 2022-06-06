.. _KubernetesIntegration:

**********************************
Kubernetes 지원
**********************************

`Kubernetes <https://kubernetes.io/>`_ 는 컨테이너화된 워크로드와 서비스를
관리하는 오픈소스 플랫폼이다. FuriosaAI SDK는 Kubernetes 환경 지원을 위해 다음 컴포넌트를 제공한다.

* FuriosaAI NPU Device Plugin (`Kubernetes Device Plugin 소개 <https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/>`_)
* FuriosaAI NPU Feature Discovery (`Node Feature Discovery 소개 <https://kubernetes-sigs.github.io/node-feature-discovery/stable/get-started/index.html>`_)

위 두 컴포넌트는 다음 기능을 제공한다.

* 노드에 가용한 NPU를 Kubernetes 클러스터가 인식하게 한다.
* Kubernetes의 ``spec.containers[].resources.limits`` 를 통해 Pod 워크로드 배포 시 NPU를 함께 스케쥴링 하게 한다.
* NPU가 장착된 머신의 NPU의 정보를 파악하여 노드의 레이블로 등록한다 (이 정보와 ``nodeSelector`` 등을 사용하면 Pod을 선택적으로 스케쥴링할 수 있다).

  * node-feature-discovery가 클러스터에 설치되어 있어야 하며, NPU가 장착된 노드에 ``nfd-worker`` Pod이 실행되고 있어야 한다.

Kubernetes 지원을 위한 셋업 과정은 다음 순서를 따라 진행하면 된다.

1. NPU 노드 준비
========================================
Kubernetes 노드의 요구 사항은 다음과 같다.

* Ubuntu 18.04, 20.04 또는 상위 버전
* Intel 호환 CPU

또한, NPU가 장착된 Kubernetes의 각 Node에 NPU 드라이버와 toolkit을 설치해야 한다.
APT 서버가 셋업되어 있다면 (:ref:`SetupAptRepository` 참고) 다음과 같이 간단히 설치할 수 있다.

.. code-block:: sh

  apt-get update && apt install -y furiosa-driver-pdma furiosa-toolkit


위 필수 패키지가 설치되면 furiosa-toolkit에 포함된 furiosactl 커맨드로 아래와 같이 NPU 인식을 확인해 볼 수 있다.
만약 아래 커맨드로 NPU가 인식되지 않는다면 환경에 따라 재부팅 후에 다시 시도해본다.

.. code-block:: sh

  $ furiosactl info
  +------+------------------+-------+--------+--------------+---------+
  | NPU  | Name             | Temp. | Power  | PCI-BDF      | PCI-DEV |
  +------+------------------+-------+--------+--------------+---------+
  | npu0 | FuriosaAI Warboy |  40°C | 1.37 W | 0000:01:00.0 | 509:0   |
  +------+------------------+-------+--------+--------------+---------+

2. Node Feature Discovery 설치
=========================================
Kubernetes에서 NPU를 활용하기 위해서는 Node Feature Discovery가 필요하다.
아래 예제처럼 커맨드를 실행하여 ``feature.node.kubernetes.io/...`` 로 시작되는 노드 레이블이 있다면 Node Feature Discovery의 DaemonSet이 이미 설치된 것으로 볼 수 있다.

.. code-block:: sh

  $ kubectl get no -o json | jq '.items[].metadata.labels'
  {
    "beta.kubernetes.io/arch": "amd64",
    "beta.kubernetes.io/os": "linux",
    "feature.node.kubernetes.io/cpu-cpuid.ADX": "true",
    "feature.node.kubernetes.io/cpu-cpuid.AESNI": "true",
    ...

* 만약 Node Feature Discovery가 설치되어 있지 않다면 다음 문서를 참조하여 설치할 수 있다.

  * `Quick start / Installation <https://kubernetes-sigs.github.io/node-feature-discovery/v0.11/get-started/quick-start.html#installation>`_ 

* Node Feature Discovery가 없어도 다음 단계를 진행할 수 있지만, 각 컴포넌트의 DaemonSet 생성 시 nodeSelector 조건을 변경해야 정상 설치가 가능하다.

3. Device Plugin, NPU Feature Discovery 설치
==============================================

NPU 노드 준비가 완료되면, Device Plugin과 NPU Feature Discovery의 DaemonSet을 다음과 같이 설치한다.

.. code-block:: sh

  kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/v0.8.0/kubernetes/deployments/device-plugin.yaml
  kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/v0.8.0/kubernetes/deployments/npu-feature-discovery.yaml

위 커맨드를 실행하고 난 뒤에 ``kubectl get daemonset -n kube-system`` 명령으로 설치한 DaemonSet이 정상 동작하는지 확인할 수 있다.
참고로 이 DaemonSet들은 NPU가 장착된 노드에만 배포되며 이를 위해 Node Feature Discovery가 각 node에 붙여주는 ``feature.node.kubernetes.io/pci-1ed2.present=true`` 정보를 사용한다.

.. code-block:: sh

  $ kubectl get daemonset -n kube-system
  NAME                           DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                                      AGE
  furiosa-device-plugin          3         3         3       3            3           feature.node.kubernetes.io/pci-1ed2.present=true   128m
  furiosa-npu-feature-discovery  3         3         3       3            3           feature.node.kubernetes.io/pci-1ed2.present=true   162m

NPU Feature Discovery가 노드에 레이블로 붙여주는 메타데이터는 다음 표와 같다.

.. _K8sNodeLabels:

.. list-table:: NPU Node Labels
   :widths: 50 50 50
   :header-rows: 1

   * - 레이블(Label)
     - 값(Value)
     - 설명(Description)
   * - beta.furiosa.ai/npu.count
     - 1
     - 해당 노드에 장착된 NPU의 수
   * - beta.furiosa.ai/npu.product
     - warboy, warboyB0
     - NPU 제품명(코드)
   * - beta.furiosa.ai/npu.family
     - warboy, renegade
     - NPU 아키텍쳐(Family)
   * - beta.furiosa.ai/machine.vendor
     - (depends on machine)
     - 머신의 제조사
   * - beta.furiosa.ai/machine.name
     - (depends on machine)
     - 머신의 제품명(코드)
   * - beta.furiosa.ai/driver.version
     - 1.3.0
     - NPU Device Driver의 버전
   * - beta.furiosa.ai/driver.version.major
     - 1
     - NPU Device Driver의 버전 중 major 파트
   * - beta.furiosa.ai/driver.version.minor
     - 3
     - NPU Device Driver의 버전 중 minor 파트
   * - beta.furiosa.ai/driver.version.patch
     - 0
     - NPU Device Driver의 버전 중 patch 파트
   * - beta.furiosa.ai/driver.reference
     - 57ac7b0
     - NPU Device Driver 빌드의 commit hash


노드의 레이블을 확인하고 싶다면 ``kubectl get nodes --show-labels`` 명령을 실행하면 된다.
다음과 같이 ``beta.furiosa.ai`` 로 시작하는 레이블이 보이면 정상적으로 설치된 것이다.

.. code-block:: sh

  kubectl get nodes --show-labels

  warboy-node01     Ready   <none>  65d   v1.20.10   beta.furiosa.ai/npu.count=1,beta.furiosa.ai/npu.product=warboy...,kubernetes.io/os=linux
  warboy-node02     Ready   <none>  12d   v1.20.10   beta.furiosa.ai/npu.count=1,beta.furiosa.ai/npu.product=warboy...,kubernetes.io/os=linux


4. NPU와 함께 Pod 배포
====================================

NPU를 Pod에 할당하기 위해서는 ``spec.containers[].resources.limits`` 에 아래와 같이 추가한다.

.. code-block:: yaml

  resources:
    limits:
      beta.furiosa.ai/npu: "1" # requesting 1 NPU


Pod 생성을 위한 `전체 예제 <https://github.com/furiosa-ai/furiosa-sdk/blob/v0.8.0/kubernetes/deployments/pod-example.yaml>`_ 는 다음과 같다.

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

Pod 생성 뒤에는 다음과 같이 NPU 할당을 확인해볼 수 있다.

.. code-block:: sh

  $ kubectl get pods npu-pod -o yaml | grep beta.furiosa.ai/npu
      beta.furiosa.ai/npu: "1"
      beta.furiosa.ai/npu: "1"


다수의 NPU 장치가 있는 노드에서 Pod을 생성했을 때, 어떤 장치가 할당되었는지는 아래와 같이 확인할 수 있다.
(SDK의 어플리케이션은 자동으로 할당된 NPU 장치를 인식한다.)

.. code-block:: sh

  $ kubectl exec npu-pod -it -- /bin/bash
  root@npu-pod:/# echo $NPU_DEVNAME
  npu0pe0-1


Pod 안에 furiosa-toolkit을 설치하면 아래처럼 furiosactl 커맨드를 이용하여 더 자세한 장치 정보를
확인할 수 있다. APT를 이용한 설치 방법은 :ref:`SetupAptRepository` 찾을 수 있다.

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
