.. _KubernetesIntegration:

**********************************
Kubernetes 지원
**********************************

`Kuberentes <https://kubernetes.io/>`_ 는 컨테이너화된 워크로드와 서비스를
관리하는 오픈소스 플랫폼이다.

FuriosaAI는 현재 가장 인기 있는 Kubernetes 플랫폼을 잘 지원하기 위해 지속적으로
개발하고 있으며 현재 제공되는 도구는 다음과 같다.

* `Kubernetes 장치 플러그인(Device Plugin) <https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/>`_ : NPU가 장착된 머신에서 유휴 NPU를 인식하고 Kubernetes에 등록하여
  Kubernetes의 ``spec.containers[].resources.limits`` 등을 이용하여 필요한 만큼의 NPU 를 스케쥴링 하게 한다.
* Kubernetes 노드 레이블러(Node Labeller): NPU가 장착된 머신의 NPU의 정보를 파악하여 노드의 레이블로 등록한다.
  이 정보를 이용하여 Pod을 선택적으로 스케쥴링할 수 있다.

위 두 컴포넌트를 이용하면 사용자는 컨테이너화된 워크로드와 서비스에 원하는 개수의 NPU를 할당하여 배포할 수 있게 된다.


NPU 노드 준비
====================================

Kubernetes 클러스터가 NPU를 인식하도록 장치 플러그인과 노드 레이블러 데몬셋(daemonset)을 설치한다.

.. code-block:: sh

  kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/v0.5.0/kubernetes/deployments/node-labeller.yaml
  kubectl apply -f https://raw.githubusercontent.com/furiosa-ai/furiosa-sdk/v0.5.0/kubernetes/deployments/device-plugin.yaml


``kubectl get daemonset -n kube-system`` 명령으로 설치한 데몬셋이 정상 동작하는지 확인할 수 있다.

.. code-block:: sh

  $ kubectl get daemonset -n kube-system
  NAME                     DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                                               AGE
  furiosa-npu-labeller     6         6         6       6            6           kubernetes.io/os=linux                                      321d
  furiosa-npu-plugin       2         2         2       2            2           alpha.furiosa.ai/npu.family=warboy,kubernetes.io/os=linux   159d

노드 레이블러(``furiosa-npu-labeller``)가 붙이는 메타데이터는 다음 표와 같다.

.. _K8sNodeLabels:

.. list-table:: NPU Node Labels
   :widths: 50 50 50
   :header-rows: 1

   * - 레이블(Label)
     - 값(Value)
     - 설명(Description)
   * - alpha.furiosa.ai/npu.family
     - warboy, renegade
     - Chip 종류
   * - alpha.furiosa.ai/npu.hwtype
     - HW 유형
     - haps, u250 (FPGA 샘플의 경우)


노드의 레이블을 확인하기 위해 ``kubectl get nodes --show-labels`` 명령을 실행하면
다음과 같은 레이블이 보여야 정상적으로 설치된 것이다.

.. code-block:: sh

  kubectl get nodes --show-labels

  warboy-node01     Ready   <none>  65d   v1.20.10   alpha.furiosa.ai/npu.family=warboy,alpha.furiosa.ai/npu.hwtype=haps...,kubernetes.io/os=linux
  warboy-node02     Ready   <none>  12d   v1.20.10   alpha.furiosa.ai/npu.family=warboy,alpha.furiosa.ai/npu.hwtype=haps...,kubernetes.io/os=linux


NPU와 함께 Pod 배포
====================================

NPU를 Pod에 할당하기 위해서는 ``spec.containers[].resources.limits`` 에 아래와 같이 추가한다.

.. code-block:: yaml

  resources:
    limits:
      furiosa.ai/npu: "1" # requesting 1 NPU


Pod 생성을 위한 전체 예제는 다음과 같다.

.. code-block:: sh

  $ cat > npu-pod.yaml <<EOL
  apiVersion: v1
  kind: Pod
  metadata:
    name: npu-pod
  spec:
    nodeSelector:
      alpha.furiosa.ai/npu.family: warboy
    containers:
      - name: linux
        image: ubuntu:focal
        command: ["/bin/sleep", "3650d"]
        resources:
          limits:
            alpha.furiosa.ai/npu: "1"
  EOL

  $ kubectl apply -f npu-pod.yaml

Pod 생성 뒤에는 다음과 같이 NPU 할당을 확인해볼 수 있다.

.. code-block:: sh

  $ kubectl exec npu-pod -it -- /bin/bash
  root@npu-pod:/# echo $NPU_DEVNAME
  npu0pe0-1
  root@npu-pod:/# furiosactl
  furiosactl controls the FURIOSA NPU.

  Find more information at: https://furiosa.ai/

  Basic Commands:
    version    Print the furiosactl version information
    list       Display NPU device list
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