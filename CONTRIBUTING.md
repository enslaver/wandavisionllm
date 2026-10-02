# Contributing

Thanks for looking. This started as one person's setup, so the most useful contributions are the
ones that make it work on *your* Mac without editing code: settings that are still hard-coded, docs
that assume too much, hooks that misfire on an agent I don't use.

## Before you start

- **Bugs:** open an issue with your Mac, macOS and component versions, and the relevant lines from
  Wanda's log tabs (strip keys and hostnames).
- **Bigger changes** (a new hook, a new tier, a different runtime): open an issue first so we can
  agree on the shape.

## Setup

```bash
git clone https://github.com/enslaver/wandavisionllm.git && cd wandavisionllm
make test     # unit tests, no network, no LiteLLM install needed
make check    # render templates, parse configs, stdlib-only and home-path checks
make lint     # ruff + a Python 3.9 compile (macOS system Python)
```

You need [uv](https://docs.astral.sh/uv/). Nothing else is installed for development.

## Rules the code follows

- **Deployed files are Python 3.9, standard library only.** That's macOS's `/usr/bin/python3`, and it
  keeps LiteLLM's environment untouched. The hooks may import `litellm` and `yaml` (both ship with
  LiteLLM). `scripts/check_repo.py` enforces this.
- **No machine-specific values in tracked files.** Home paths are `__HOME__` (config files) or `$HOME`
  (scripts); hostnames and endpoints are settings. CI rejects `/Users/<name>/`.
- **Secrets never enter the repo.** They live in `~/.litellm/env` and are referenced as
  `os.environ/NAME` in `config.yaml`.
- **A new deployed file** goes in the folder that mirrors its destination and into `COMPONENTS` in
  `deploy.py`.
- **Hooks never break requests.** Wrap new logic so an exception logs and passes the request through.
  Anything that changes behavior gets a `shadow` mode first.
- **Match the surrounding code:** short functions, comments that explain *why*, no new abstractions
  for one caller.

## Tests

- Unit tests live next to the hooks (`litellm/test_*.py`). Add one for every behavior change; build
  requests with the helpers already in those files.
- Routing, model or vision changes: also run the live suite against a running stack and paste its
  summary in the PR: `cd litellm && uvx --with pyyaml python3 suite.py`.
- Loop-breaker threshold changes: replay real transcripts with `python3 litellm/replay.py DIR` and
  report how many legitimate runs would now trip.

## Pull requests

- One topic per PR; describe the problem before the fix.
- Add a line under `[Unreleased]` in `CHANGELOG.md`.
- CI must pass.

## Releases

Maintainers: move `[Unreleased]` to a new version section in `CHANGELOG.md`, bump `version` in
`pyproject.toml`, commit, then `git tag vX.Y.Z && git push --tags`. The release workflow checks both,
runs the tests and publishes the GitHub release with that section as its notes.
