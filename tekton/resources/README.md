# Kubernetes resources for SDK CI

### Secrets

Secrets used by SDK CI are managed by [Sealed secret](https://github.com/bitnami-labs/sealed-secrets).

To seal original secrets, (if you modify original secret.yaml)

```bash
csplit -z -s secret.yaml -f secret- -b %02d.yaml '/^---$/' '{*}'

for i in secret-0[0-9]*.yaml; do kubeseal -f $i -w sealed-$i; done
```
