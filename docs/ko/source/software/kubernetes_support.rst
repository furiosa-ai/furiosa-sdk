.. _KubernetesIntegration:

**********************************
Kubernetes 지원
**********************************

`Kubernetes <https://kubernetes.io/>`_ 는 컨테이너화된 워크로드와 서비스를
관리하는 오픈소스 플랫폼이다. FuriosaAI SDK는 Kubernetes 환경 지원을 위해 다음 컴포넌트를 제공한다.

* FuriosaAI NPU Device Plugin (`Kubernetes Device Plugin 소개 <https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/>`_)
* FuriosaAI NPU Feature Discovery (`Node Feature Discovery 소개 <https://kubernetes-sigs.github.io/node-feature-discovery/stable/get-started/index.html>`_)
* FuriosaAI NPU Metrics Exporter

위의 컴포넌트는 다음 기능을 제공한다.

* 노드에 가용한 NPU를 Kubernetes 클러스터가 인식하게 한다.
* Kubernetes의 ``spec.containers[].resources.limits`` 를 통해 Pod 워크로드 배포 시 NPU를 함께 스케쥴링 하게 한다.
* NPU가 장착된 머신의 NPU의 정보를 파악하여 노드의 레이블로 등록한다 (이 정보와 ``nodeSelector`` 등을 사용하면 Pod을 선택적으로 스케쥴링할 수 있다).

  * node-feature-discovery가 클러스터에 설치되어 있어야 하며, NPU가 장착된 노드에 ``nfd-worker`` Pod이 실행되고 있어야 한다.

* 노드에 장착된 NPU의 상태 정보를 Prometheus에서 수집할 수 있게 한다.

Kubernetes 지원을 위한 셋업 과정은 다음 순서를 따라 진행하면 된다.

1. NPU 노드 준비
========================================
Kubernetes 노드의 요구 사항은 다음과 같다.

* Ubuntu 20.04 또는 상위 버전
* Intel 호환 CPU

또한, NPU가 장착된 Kubernetes의 각 Node에 NPU 드라이버와 toolkit을 설치해야 한다.
APT 서버가 셋업되어 있다면 (:ref:`SetupAptRepository` 참고) 다음과 같이 간단히 설치할 수 있다.

.. code-block:: sh

  apt-get update && apt install -y furiosa-driver-warboy furiosa-toolkit


위 필수 패키지가 설치되면 furiosa-toolkit에 포함된 furiosactl 커맨드로 아래와 같이 NPU 인식을 확인해 볼 수 있다.
만약 아래 커맨드로 NPU가 인식되지 않는다면 환경에 따라 재부팅 후에 다시 시도해본다.

.. code-block:: sh

  $ furiosactl info
  +------+------------------+-------+--------+--------------+---------+
  | NPU  | Name             | Temp. | Power  | PCI-BDF      | PCI-DEV |
  +------+------------------+-------+--------+--------------+---------+
  | npu0 | FuriosaAI Warboy |  40°C | 1.37 W | 0000:01:00.0 | 509:0   |
  +------+------------------+-------+--------+--------------+---------+

.. _SetupNodeFeatureDiscovery:

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

* Node Feature Discovery 실행 시 다음 옵션이 적용되어 있어야 한다.

  * ``nfd-master`` 의 ``--extra-label-ns`` 옵션에 ``beta.furiosa.ai`` 가 포함되어 있어야 함
  * ``nfd-worker`` 의 config 파일에

    * ``sources.pci.deviceLabelFields`` 의 값에 ``vendor`` 만 있어야 함
    * ``sources.pci.deviceClassWhitelist`` 의 값 중 ``"12"`` 가 포함되어 있어야 함

.. code-block::
  :caption: nfd-worker.conf

  sources:
    pci:
      deviceClassWhitelist:
      - "02"
      - "0200"
      - "0207"
      - "0300"
      - "0302"
      - "12"
      deviceLabelFields:
      - vendor

.. note::

  Node Feature Discovery는 필수 컴포넌트가 아니지만 설치를 권장한다. 미사용 시 수행해야 하는 추가 작업에 대해서는 다음 단계에서 설명한다.


.. _InstallingDevicePluginAndNfd:

3. Device Plugin, NPU Feature Discovery, NPU Metrics Exporter 설치
=====================================================================

NPU 노드 준비가 완료되면, Device Plugin, NPU Feature Discovery와 NPU Metrics Exporter의 DaemonSet을 다음과 같이 설치한다.

.. code-block:: sh

  kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/0.8.0/kubernetes/deployments/device-plugin.yaml
  kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/0.8.0/kubernetes/deployments/npu-feature-discovery.yaml
  kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/0.8.0/kubernetes/deployments/npu-metrics-exporter.yaml
  

위 커맨드를 실행하고 난 뒤에 ``kubectl get daemonset -n kube-system`` 명령으로 설치한 DaemonSet이 정상 동작하는지 확인할 수 있다.
참고로 이 DaemonSet들은 NPU가 장착된 노드에만 배포되며 이를 위해 Node Feature Discovery가 각 node에 붙여주는 ``feature.node.kubernetes.io/pci-1ed2.present=true`` 정보를 사용한다.

.. code-block:: sh

  $ kubectl get daemonset -n kube-system
  NAME                           DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                                      AGE
  furiosa-device-plugin          3         3         3       3            3           feature.node.kubernetes.io/pci-1ed2.present=true   128m
  furiosa-npu-feature-discovery  3         3         3       3            3           feature.node.kubernetes.io/pci-1ed2.present=true   162m
  furiosa-npu-metrics-exporter   3         3         3       3            3           feature.node.kubernetes.io/pci-1ed2.present=true   162m

만약 :ref:`단계 2<SetupNodeFeatureDiscovery>` 에서 Node Feature Discovery 설치를 생략하였다면 NPU Feature Discovery는 설치가 불필요하며, 나머지 컴포넌트는 다음 절차를 추가로 수행한 후 설치가 가능하다.

* 위에서 제시한 YAML 파일을 수정하여 DaemonSet의 nodeSelector 조건을 변경하고 설치해야 함

  * 컴포넌트를 클러스터 내의 모든 노드에 설치할 경우

    * ``feature.node.kubernetes.io/pci-1ed2.present: "true"`` 를 제거

  * 컴포넌트를 클러스터 내에서 NPU가 설치된 일부 노드에 설치할 경우

    * 해당하는 노드에 label을 추가 (예, ``kubectl label node <nodename> furiosa=true`` )
    * nodeSelector 조건을 변경 (예, ``feature.node.kubernetes.io/pci-1ed2.present: "true"`` 대신 ``furiosa: "true"`` )


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


Device Plugin 설정
--------------------------------------
Device Plugin의 실행 옵션은 명령행의 인자로 지정하거나 설정 파일을 통해 지정할 수 있도록 두 가지 방법을 제공한다.

1. 명령행 입력 방식

``k8s-device-plugin`` 명령을 실행하면서 인자를 통해 옵션을 지정할 수 있다.

.. code-block:: sh

  $ k8s-device-plugin --interval 10

Pod 또는 DaemonSet 명세에서는 다음과 같이 명령행 인자를 설정 할 수 있다.

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
  # (이하 생략)

.. list-table:: k8s-device-plugin 인자 목록
   :widths: 50 150 50
   :header-rows: 1

   * - 항목
     - 설명
     - 기본값
   * - default-pe
     - Pod 할당 시 기본값으로 적용되는 Core 유형 (Fusion/Single)
     - Fusion
   * - interval
     - 장치 탐색 주기 (단위: 초)
     - 10
   * - disabled-devices
     - 할당 대상에서 제외할 장치 지정(콤마로 여러 장치를 지정 가능)
     - 
   * - plugin-dir
     - kubelet의 device-plugin 디렉토리 경로
     - /var/lib/kubelet/device-plugins
   * - socket-name
     - <plugin-dir> 아래에 생성할 socket 파일의 이름
     - furiosa-npu
   * - resource-name
     - k8s 노드에 등록할 NPU 자원의 이름
     - beta.furiosa.ai/npu

2. 설정파일 지정 방식

``k8s-device-plugin`` 명령을 실행하면서 ``config-file`` 인자를 통해 설정 파일을 지정할 수 있다.
단, ``config-file`` 을 지정한 경우 나머지 인자들은 사용할 수 없다.

.. code-block:: sh

  $ k8s-device-plugin --config-file /etc/furiosa/device-plugin.conf

.. code-block:: yaml
   :caption: /etc/furiosa/device-plugin.conf

   interval: 10
   defaultPe: Fusion
   disabledDevices:             # warboy-node01 노드의 npu1 장치를 사용하지 않음을 의미
     - devName: npu1
       nodeName: warboy-node01
   pluginDir: /var/lib/kubelet/device-plugins
   socketName: furiosa-npu
   resourceName: beta.furiosa.ai/npu

설정 파일은 Yaml 포맷의 텍스트 형태이다. 파일 내용이 변경되면 변경 사항이 Device Plugin에 즉시 적용된다. 설정이 업데이트 되었음은 Device Plugin의 로그를 통해 확인할 수 있다.
(단, ``pluginDir`` , ``socketName``, ``resourceName`` 이 항목들의 변경을 적용하기 위해서는 재시작이 필요하다.)


:ref:`InstallingDevicePluginAndNfd` 의 설치에서 제공하는 ``device-plugin.yaml`` 는 기본적으로 ConfigMap 기반의 설정 파일을 사용하는 구성을 제공한다.
만약 Device Plugin의 실행 옵션을 변경하고 싶다면 이 ConfigMap을 수정하고, 변경된 ConfigMap이 Pod에 반영되면 Device Plugin은 이를 읽고 변경사항을 적용한다.

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


4. NPU와 함께 Pod 배포
====================================

NPU를 Pod에 할당하기 위해서는 ``spec.containers[].resources.limits`` 에 아래와 같이 추가한다.

.. code-block:: yaml

  resources:
    limits:
      beta.furiosa.ai/npu: "1" # requesting 1 NPU


Pod 생성을 위한 `전체 예제 <https://github.com/furiosa-ai/furiosa-sdk/blob/0.8.0/kubernetes/deployments/pod-example.yaml>`_ 는 다음과 같다.

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

5. NPU 모니터링
====================================

``npu-metrics-exporter`` 를 설치하면 DaemonSet과 Service가 생성된다.
DaemonSet을 통해 실행되는 Pod에서는 NPU의 각종 상태 정보를 지표로 출력하여 모니터링에 도움이 되는 정보를 제공한다.
지표 정보는 Prometheus 형식으로 표현되며, Kubernetes 클러스터내에 service discovery가 활성화 된 Prometheus가 설치되어 있다면
Prometheus가 Exporter를 통해 출력되는 데이터를 자동으로 수집한다.

수집된 데이터는 Grafana 등의 시각화 도구를 통해 확인할 수 있다.

.. list-table:: npu-metrics-exporter 수집 항목 목록
   :widths: 250 250
   :header-rows: 1

   * - 이름
     - 설명
   * - furiosa_npu_alive
     - NPU 동작 상태 (1:정상)
   * - furiosa_npu_uptime
     - NPU 동작 시간 (s)
   * - furiosa_npu_error
     - NPU에서 감지된 에러의 수
   * - furiosa_npu_hw_temperature
     - NPU의 컴포넌트 별 온도 (°mC)
   * - furiosa_npu_hw_power
     - NPU의 순간 전력사용량 (µW)
   * - furiosa_npu_hw_voltage
     - NPU의 순간 전압 (mV)
   * - furiosa_npu_hw_current
     - NPU의 순간 전류 (mA)
