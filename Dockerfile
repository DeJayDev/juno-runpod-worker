# Use specific version of nvidia cuda image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Labeling for GHCR, then we'll build the image :)
LABEL org.opencontainers.image.source https://github.com/dejaydev/juno-runpod-worker
LABEL org.opencontainers.image.description "juno runpod image"
LABEL org.opencontainers.image.licenses MIT

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends libcudnn8 python3-pip python-is-python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/*.list

# Install Python dependencies (Worker Template)
COPY juno/builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Copy source code into image
COPY juno .

# Set default command
CMD ["python", "-m", "juno.handler", "--rp_serve_api"]
