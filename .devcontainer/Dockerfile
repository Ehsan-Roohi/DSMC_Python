FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       gfortran \
       python3 \
       python3-numpy \
       ca-certificates \
       coreutils \
       gzip \
       tar \
    && rm -rf /var/lib/apt/lists/*

USER vscode
