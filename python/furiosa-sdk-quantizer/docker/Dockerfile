FROM python:3.7.9

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq
RUN apt-get install -qq \
    build-essential \
    cmake \
    protobuf-compiler \
    libprotoc-dev
