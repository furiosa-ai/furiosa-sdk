# SDK CI

Furiosa SDK uses [Tekton](https://tekton.dev/) for CI. This directory contains related manifests to install and run CI.

### Installation

This steps explains initial setups for Furiosa SDK CI. We assumes that you have Kubernetes cluster already.

1. Create namespace `ci-furiosa-sdk`

```sh
kubectl create namespace ci-furiosa-sdk 
```

2. Create required resources: `Secret`, `ServiceAccount`... via Kustomize

```sh
kubectl apply -k ./resources
```

### Run Pipeline 

You can run the Pipeline manually use `tkn` CLI.

```sh
tkn pipeline start \
    --filename ./pipeline.yaml \
    --serviceaccount build-bot \
    --showlog \
    --workspace name=source,volumeClaimTemplateFile=workspace-template.yaml \
    --workspace name=conda,volumeClaimTemplateFile=workspace-template.yaml \
    --workspace name=apt-credential,secret=apt-credential \
    --workspace name=pypi-credential,secret=pypi-credential \
    --pod-template ./pod-template.yaml \
    --use-param-defaults \
    --pipeline-timeout 1h30m \
    --namespace ci-furiosa-sdk
```

- --*serviceaccount*: To inject secrets
- --*showlog*: To display logs
- --*workspace*: To pass workspace parameter. See https://github.com/tektoncd/cli/blob/main/docs/cmd/tkn_pipeline_start.md
- --*use-param-defaults*: To use default parameter if not specified
- --*param key=value*: To specify parameters
- --*pipeline-timeout*: To specify pipeline level timeout 
- --*pod-template**: To specify NPU Pod affinity
