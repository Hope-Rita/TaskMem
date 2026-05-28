#!/usr/bin/env bash
#
# Install the heavyweight VL stack (vLLM + Megatron + flash-attn + transformer
# engine) for Qwen3-VL training / inference.
#
# This script targets CUDA 12.8 + PyTorch 2.8 + Python 3.10. Adjust the wheel
# URLs below if your environment differs.
#
# Prerequisite: run `bash setup.sh` first. This script does not re-install the
# lightweight Python deps (insightface / hdbscan / openai / ...) or the
# `ffmpeg` system package that the pipeline needs.
set -e

sudo pip install --no-cache-dir \
    compressed-tensors==0.11.0 \
    frozendict==2.4.6 \
    lm-format-enforcer==0.11.3 \
    openai==1.99.1 \
    openai-harmony==0.0.4 \
    outlines_core==0.2.11 \
    xformers==0.0.32.post1 \
    xgrammar==0.1.25 \
    qwen-vl-utils==0.0.14 \
    tokenizers==0.22.1 \
    transformers==4.57.1 \
    uvloop==0.21.0 \
    flashinfer-python==0.2.2

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    sudo pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128
    sudo pip install -v --no-build-isolation 'transformer_engine[pytorch]==2.8.0'
    wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.18/flash_attn-2.7.4+cu128torch2.8-cp310-cp310-linux_x86_64.whl
    sudo pip install --no-cache-dir flash_attn-2.7.4+cu128torch2.8-cp310-cp310-linux_x86_64.whl
    rm flash_attn-2.7.4+cu128torch2.8-cp310-cp310-linux_x86_64.whl
    sudo pip install vllm==0.11.0 --no-deps
    sudo pip install -U git+https://github.com/ISEEKYAN/mbridge.git
    sudo pip install --no-deps --no-cache-dir git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
    pip install "numpy<2.0.0"
else
    sudo pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
    pip install "numpy<2.0.0"
fi

pip install moviepy==2.1.2
pip install pydub
pip install opencv-python-headless==4.10.0.84
pip install json-repair
pip install httpx==0.23.3

if [ -n "${VLLM_PLUGINS:-}" ]; then
    pip install -e .
fi
