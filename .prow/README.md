# Prow for SDK

Furiosa SDK uses [Prow](https://docs.prow.k8s.io/docs/) to trigger CI jobs.

### Getting started

We assumed you already have Prow, and this repository only have prow configuration `.prow.yaml`

We use modified [tkn-watch](https://github.com/ileixe/tkn-watch) to get exit code from Tekton PipelineRun.

#### Build container image

From [Dockerfile](https://github.com/ileixe/tkn-watch/blob/withlog/Dockerfile)

```
docker build . --tag ileixe/tkn-watch:69b5bbb-v941edb36 && docker push ileixe/tkn-watch:69b5bbb-v941edb36
```

`69b5bbb` is commit revision for the [tkn-watch](https://github.com/ileixe/tkn-watch) and `v941edb36a` is for [tkn cli](https://github.com/tektoncd/cli)

This container image does not have entrypoint (command in Kubernetes), and we use `ConfigMap` to mount the entrypoint. Apply `configmap.yaml` when you update the entrypoint.


```sh
kubectl apply -f configmap.yaml -n test-pods
```

#### Run Tekton Pipeline

The Pod will runs the `tkn` command:

See [README.md](../tekton/README.md) to understand each parameter.

```bash
/usr/bin/tkn pipeline start --serviceaccount build-bot --filename tekton/pipeline.yaml --workspace name=apt-credential,secret=apt-credential --workspace name=pypi-credential,secret=pypi-credential --workspace name=source,volumeClaimTemplateFile=tekton/workspace-template.yaml --workspace name=conda,volumeClaimTemplateFile=tekton/workspace-template.yaml --pod-template tekton/pod-template.yaml --use-param-defaults --namespace ci-furiosa-sdk
```
