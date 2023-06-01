#!/bin/bash

mkdir -p python

for MODULE in "common" "runtime" "optimizer" "quantizer" "registry" "serving" "server"; do
  sphinx-apidoc -f --implicit-namespaces --extensions 'sphinx.ext.napoleon' --no-toc -o python ../../../../python/furiosa-${MODULE}/furiosa
done

# remove undocumented `furiosa` and `furiosa.sdk` modules, and
# put our own `modules.rst` to include all submodules of the `furiosa` namespace package
rm -f python/furiosa.rst python/furiosa.sdk.rst
cp modules.rst python
