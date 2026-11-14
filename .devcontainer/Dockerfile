FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Node 20 for the frontend toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
