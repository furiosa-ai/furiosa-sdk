#!/bin/bash

SCRIPT_PATH=$(readlink -f "$0")
B=$(dirname "$SCRIPT_PATH")/..

cd $B/examples/inferences
./image_classify.py ../assets/images/car.jpg && \
./image_segmentation.py ../assets/images/segmentation_1.jpg

cd $B
./examples/profiler/simple.py && \
./examples/profiler/temporary_disable.py && \
./examples/profiler/trace_with_dataframe.py
