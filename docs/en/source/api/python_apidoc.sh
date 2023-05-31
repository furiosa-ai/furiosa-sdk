#!/bin/bash

mkdir -p python

for MODULE in "sdk" "common" "runtime" "optimizer" "quantizer" "registry" "serving" "server"; do
  sphinx-apidoc --implicit-namespaces --extensions 'sphinx.ext.napoleon' -o python ../../../../python/furiosa-${MODULE}/furiosa
done

cp modules.rst python
