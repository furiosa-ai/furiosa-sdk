#!/bin/bash

sphinx-apidoc --implicit-namespaces -o python ../../../../python/furiosa-runtime/furiosa
sphinx-apidoc --implicit-namespaces -o python ../../../../python/furiosa-quantizer/furiosa
sphinx-apidoc --implicit-namespaces -o python ../../../../python/furiosa-models/furiosa
