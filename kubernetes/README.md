# FuriosaAI NPU support in Kubernetes'
## FuriosaAI NPU Device Plugin
### Introduction
This is a Kubernetes [Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) implementation 
that enables the resource management of FuriosaAI NPU in your kubernetes cluster 
consisting of nodes equipped with FuriosaAI NPU hardware.
With this plugin, you will be able to run jobs that require FuriosaAI NPU.

### Installation
We provide Kubernetes [DaemonSet](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/) to deploy 
easily the Device Plugin to all machines which are equipped with Furiosa NPU device. 
This repository includes the pre-defined DaemonSet yaml file named `device-plugin.yaml`.
After downloading this yaml file, please run as following:
```sh
$ kubectl apply -f device-plugin.yaml
```

You can make sure that the plugin is working well if you run the command ```kubectl describe nodes``` as following:   
```
$ kubectl describe nodes demo01
Name:               demo01
Roles:              <none>
Labels:             alpha.furiosa.ai/npu.family=warboy
                    beta.kubernetes.io/os=linux
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=demo01
                    kubernetes.io/os=linux
Annotations:        kubeadm.alpha.kubernetes.io/cri-socket: /var/run/dockershim.sock
...
Taints:             <none>
Unschedulable:      false
Addresses:
  InternalIP:  10.1.0.2
  Hostname:    demo01
Capacity:
  alpha.furiosa.ai/npu:  1
...
Allocatable:
  alpha.furiosa.ai/npu:  1
...
Non-terminated Pods:          (10 in total)
  Namespace              Name                                  CPU Requests  CPU Limits  Memory Requests  Memory Limits  Age
  ---------              ----                                  ------------  ----------  ---------------  -------------  ---
...
  kube-system            furiosa-device-plugin-bs9b6           0 (0%)        0 (0%)      0 (0%)           0 (0%)         3d2h
...
```

### Usage
You can specify `alpha.furiosa.ai/npu` in your container spec.

This repository includes the pre-defined Pod yaml file named `npu-pod.yaml`.
After downloading this yaml file, please run as following:
```sh
$ kubectl apply -f pod-example.yaml
$ kubectl exec -it npu-pod -- bash
```


## FuriosaAI Node Labeller
### Introduction
This program automatically labels Kubernetes nodes with NPU properties if a node has 
one or more NPU devices. You can leverage the NPU properties for scheduling 
your application which require specific FuriosaAI NPU hardwares.

### Installation 
We provide Kubernetes [DaemonSet](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/) to deploy easily the Node labellers to all Linux machines. 
This repository includes the pre-defined DaemonSet yaml file named `device-plugin.yaml`.
After downloading this yaml file, please run as following:
```sh
$ kubectl apply -f node-labeller.yaml
```

Example:
```
$ kubectl get nodes --show-labels
NAME        STATUS   ROLES    AGE   VERSION    LABELS
demo01      Ready    <none>   9d    v1.20.13   alpha.furiosa.ai/npu.family=warboy, ..
```

You can make sure that the node labeller is working well if you run the command ```kubectl describe nodes``` as following:   
```
$ kubectl describe nodes demo01
Name:               demo01
Roles:              <none>
Labels:             alpha.furiosa.ai/npu.family=warboy
                    beta.kubernetes.io/os=linux
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=demo01
                    kubernetes.io/os=linux
Annotations:        kubeadm.alpha.kubernetes.io/cri-socket: /var/run/dockershim.sock
...
Non-terminated Pods:          (10 in total)
  Namespace              Name                                  CPU Requests  CPU Limits  Memory Requests  Memory Limits  Age
  ---------              ----                                  ------------  ----------  ---------------  -------------  ---
...
  kube-system            furiosa-node-labeller-8kpk7           0 (0%)        0 (0%)      0 (0%)           0 (0%)         3d2h
...
```