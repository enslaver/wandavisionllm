#!/bin/bash
# JEV-27B-VL's POST /v1/decide server on mlx-vlm's python (pipx venv: mlx_vlm, fastapi, uvicorn).
# Usage: serve-judge.sh PORT. Not deployed: copy it and serve_decide_mlx.py to ~/.judge on the machine that runs it.
MODEL="$HOME/models/LLM/JEV-27B-VL-mlx-4bit"
exec "$HOME/Library/Application Support/pipx/venvs/mlx-vlm/bin/python" -I "$HOME/.judge/serve_decide_mlx.py" \
  --model "$MODEL" --served-model-name judge --port "$1" \
  --max-context 32768 --max-images 8
