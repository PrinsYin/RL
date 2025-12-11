#!/bin/bash
set -e

git submodule update --init --recursive

VENV_NAME=".venv_test"
CONFIG_FILE="examples/configs/grpo_math_1B_sglang.yaml"

if [ -d "$VENV_NAME" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_NAME"
fi

uv venv "$VENV_NAME"

source "$VENV_NAME/bin/activate"
uv sync --extra sglang
uv pip install "numpy<2.0"


# export CUDA_VISIBLE_DEVICES=
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
ray stop --force

python examples/run_grpo_math.py --config "$CONFIG_FILE"

