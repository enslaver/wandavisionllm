# Everything CI runs, runnable locally. Needs uv (https://docs.astral.sh/uv/).
.PHONY: all test test-lora lint check shellcheck render

all: lint test check

test:
	cd litellm && uvx --with pyyaml pytest -q -p no:cacheprovider
	cd Vision/media && uvx --with pyyaml pytest -q -p no:cacheprovider
	cd gpu-box && uvx pytest -q -p no:cacheprovider

test-lora:  ## lora/ tests need MLX (Apple Silicon): run in the lora venv (lora/README.md, Setup)
	cd lora && ~/lora/.venv/bin/python -m pytest -q -p no:cacheprovider

lint:
	uvx ruff check .
	/usr/bin/python3 -m py_compile deploy.py wanda/server.py litellm/*.py Vision/media/*.py \
		gpu-box/*.py Vision/judge/rank.py Vision/open-webui/video_tool.py lora/add_mtp.py lora/unsloth_vision.py   # macOS system Python 3.9

check:
	uv run --no-project --with pyyaml python3 scripts/check_repo.py

shellcheck:
	for f in litellm/start.sh wanda/install.sh lora/unsloth_queue.sh bili/start.sh; do bash -n $$f || exit 1; done
	for f in mtplx/bin/*.sh bin/*.sh; do zsh -n $$f || exit 1; done

render:  ## write the filled-in files to ./rendered for review
	./deploy.py render rendered
