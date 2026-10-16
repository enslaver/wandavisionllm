"""mtplx_effort.py serve ... — run `mtplx` with per-request reasoning_effort -> thinking budget.

The tier scripts start mtplx through this file (with mtplx's own runtime-venv python) instead of
~/.mtplx/bin/mtplx. It runs mtplx's normal CLI; `mtplx serve` then execs the server process, and that exec
is rerouted through this file (SERVER_FLAG), which patches two mtplx 2.12.2 functions before serving.

Why: an agent harness wants a thinking limit per call kind (planner, verdict, worker turn,
re-ask...), sent the standard way as OpenAI `reasoning_effort`. mtplx resolves reasoning_effort through
the model's reasoning codec, and the Qwen3.5/3.6 codec declares no effort levels, so the client's value
is dropped (request log: request_reasoning_effort null). The only budget mtplx has is server-wide
(--agent-thinking-budget / MTPLX_THINKING_BUDGET) and covers tool-carrying requests only.

  request_policy._control_ownership_observability  also records the client's raw reasoning_effort
                                                   in the request's observability (EFFORT_KEY)
  openai._thinking_guard_config_for_request         gives a request that sent one of EFFORT_BUDGETS a
                                                   thinking_guard budget of that size, tools or not

At the budget mtplx's thinking_guard forces its bridge line and </think>, then the model answers
(thinking counts inside max_tokens). "none" is a 1-token budget with no bridge line, not
enable_thinking=false: the thinking flag is part of mtplx's session-cache key and the effort is not,
so changing effort inside one conversation keeps the prefix cache. A request with no (or an unknown)
reasoning_effort, or with thinking off, keeps mtplx's own behaviour (the tier's server-wide budget).

MTPLX_THINKING_BUDGET would replace every budget, per-request ones included, so a tier that sets it gets
it moved to the equivalent --agent-thinking-budget flag here. The same numbers are in
litellm/ultron_admit.py (EFFORT_BUDGETS), which echoes them in x-ultron-route. Enable it per tier
with `effort = yes` in tiers.conf; the tier script must start mtplx through this file.
"""
import os
import sys

EFFORT_BUDGETS = {"none": 1, "minimal": 512, "low": 2048, "medium": 8192, "high": 16384, "xhigh": 32768,
                  "max": 32768}
EFFORT_KEY = "ultron_reasoning_effort"


def install() -> None:
    from mtplx.server import openai as srv
    from mtplx.server import request_policy as policy
    from mtplx.thinking_guard import think_marker_ids, thinking_guard_config_from_env

    record = policy._control_ownership_observability
    guard = srv._thinking_guard_config_for_request

    def control_ownership_observability(request, *args, **kwargs):
        out = record(request, *args, **kwargs)
        obs = kwargs.get("observability")
        effort = str(getattr(request, "reasoning_effort", None) or "").strip().lower()
        if isinstance(obs, dict) and effort in EFFORT_BUDGETS and kwargs.get("client_controls_allowed", True):
            obs[EFFORT_KEY] = effort
        return out

    def thinking_guard_config_for_request(state, *, prompt_ids, request_observability):
        obs = request_observability or {}
        budget = EFFORT_BUDGETS.get(obs.get(EFFORT_KEY))
        tokenizer = getattr(getattr(state, "runtime", None), "tokenizer", None)
        markers = think_marker_ids(tokenizer) if budget else None
        if not budget or not obs.get("request_enable_thinking") or markers is None:
            return guard(state, prompt_ids=prompt_ids, request_observability=request_observability)
        config = thinking_guard_config_from_env(
            True,
            budget_tokens=budget,
            tokenizer=tokenizer,
            starts_in_think=int(markers[0]) in {int(t) for t in prompt_ids[-8:]},
            bridge_text="" if budget <= 1 else None,
        )
        return config if getattr(config, "enabled", False) else None

    policy._control_ownership_observability = control_ownership_observability
    srv._thinking_guard_config_for_request = thinking_guard_config_for_request


def move_env_budget(argv: list) -> list:
    budget = os.environ.pop("MTPLX_THINKING_BUDGET", "").strip()
    if budget and "--agent-thinking-budget" not in argv:
        argv = argv + ["--agent-thinking-budget", budget]
    return argv


def serve_through_wrapper() -> None:
    """`mtplx serve` doesn't serve: it builds the server's command line and os.execvpe's
    `python -P -m mtplx.server.openai ...` (commands/public.py), so patches made in this process are gone.
    Rewrite that exec to run this file again with SERVER_FLAG, which patches and then calls the server."""
    execvpe = os.execvpe

    def exec_patched(file, args, env):
        args = list(args)
        for i in range(len(args) - 1):
            if args[i] == "-m" and args[i + 1] == "mtplx.server.openai":
                args[i:i + 2] = [os.path.abspath(__file__), SERVER_FLAG]
                break
        return execvpe(file, args, env)

    os.execvpe = exec_patched


SERVER_FLAG = "--ultron-effort-server"

if __name__ == "__main__":
    if sys.argv[1:2] == [SERVER_FLAG]:  # the server process (see serve_through_wrapper)
        argv = move_env_budget(sys.argv[2:])  # the server, unlike `mtplx serve`, has --agent-thinking-budget
        install()
        from mtplx.server import openai as server
        server.main(argv)
    else:  # `mtplx serve ...` as the tier scripts call it
        serve_through_wrapper()
        from mtplx.cli import main
        sys.argv[0] = "mtplx"
        sys.exit(main())
