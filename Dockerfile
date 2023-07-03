# syntax=docker/dockerfile:1.4.1

# Dockerfile for SDK CI
ARG DIST=focal

FROM ubuntu:$DIST

ARG HAL_VERSION="0.12.*"
ARG NUX_VERSION="0.10.*"
ARG ONNX_RUNTIME_VERSION="1.15.*"
ARG ARCHIVE=https://internal-archive.furiosa.dev

ENV DEBIAN_FRONTEND="noninteractive"

# Install basic dependencies
RUN <<EOF
    set -eu

    apt update
    apt install --no-install-recommends --assume-yes \
        git \
        ca-certificates \
        apt-transport-https \
        gnupg \
        wget \
        python3-opencv \
        gcc-aarch64-linux-gnu \
        build-essential \
        cmake

    apt-key adv --keyserver keyserver.ubuntu.com --recv-key 5F03AFA423A751913F249259814F888B20B09A7E

    rm -rf /var/lib/apt/lists/*
EOF

# Install internal dependencies
COPY <<EOF /etc/apt/sources.list.d/furiosa.list
    deb [arch=amd64] $ARCHIVE/ubuntu focal restricted
    deb [arch=amd64] $ARCHIVE/ubuntu focal-nightly restricted
EOF

RUN --mount=type=secret,id=furiosa.conf,dst=/etc/apt/auth.conf.d/furiosa.conf,required <<EOF
    set -eu

    apt update
    apt install --no-install-recommends --assume-yes \
        furiosa-libhal-warboy=$HAL_VERSION \
        furiosa-libnux=$NUX_VERSION \
        libonnxruntime=$ONNX_RUNTIME_VERSION

    rm -rf /var/lib/apt/lists/*
EOF

COPY . .
