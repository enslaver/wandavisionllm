#!/usr/bin/env python3
"""anthropic_to_trace.py — a captured /v1/messages body -> a trace (what the local tier sees).

Runs LiteLLM's own Anthropic->OpenAI adapter, then ultron_admit.normalize_history, so the trace
matches what reaches llama-swap. Needs LiteLLM's interpreter:

    ~/.local/share/uv/tools/litellm/bin/python anthropic_to_trace.py cc_req.json ~/lora/traces/name.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "litellm"))
import ultron_admit as ua  # noqa: E402
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (  # noqa: E402
    LiteLLMAnthropicMessagesAdapter,
)

src, dst = sys.argv[1], sys.argv[2]
body = json.load(open(src))
req = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
    anthropic_message_request=dict(body, model="hosted_vllm/sonnet"))
req = req[0] if isinstance(req, tuple) else req
req = req if isinstance(req, dict) else req.model_dump(exclude_none=True)
msgs = ua.normalize_history([dict(m) for m in req["messages"]], think_in_content=True)
Path(dst).expanduser().parent.mkdir(parents=True, exist_ok=True)
json.dump({"messages": msgs, "tools": req.get("tools") or []}, open(Path(dst).expanduser(), "w"), default=str)
print(f"{dst}: {len(msgs)} messages, {len(req.get('tools') or [])} tools")
