"""Serialized, auditable API calls for the fair-adapted Embodied-RAG run."""

import asyncio
import json
import os
import time
from pathlib import Path


_LOCK = asyncio.Lock()


def _usage_path():
    value = os.environ.get("EMBODIED_RAG_USAGE_LOG")
    return Path(value) if value else None


def _totals(path):
    if not path or not path.exists():
        return 0.0, 0
    cost = 0.0
    calls = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if not row.get("cached", False):
            calls += 1
            cost += float(row.get("estimated_cost_usd", 0.0) or 0.0)
    return cost, calls


async def create_completion(client, task, **kwargs):
    """Serialize calls so concurrent forest summaries cannot race past budget."""
    async with _LOCK:
        if str(kwargs.get("model", "")).startswith("qwen3"):
            kwargs.setdefault("extra_body", {"enable_thinking": False})
        path = _usage_path()
        cost, calls = _totals(path)
        max_cost = float(os.environ.get("EMBODIED_RAG_MAX_COST_USD", "1.0"))
        max_calls = int(os.environ.get("EMBODIED_RAG_MAX_API_CALLS", "3000"))
        if cost >= max_cost or calls >= max_calls:
            raise RuntimeError(
                f"Embodied-RAG API budget exhausted: cost=${cost:.6f}/{max_cost:.6f}, "
                f"calls={calls}/{max_calls}"
            )
        started = time.perf_counter()
        response = await client.chat.completions.create(**kwargs)
        usage = response.usage
        input_rate = float(os.environ.get("EMBODIED_RAG_INPUT_USD_PER_MILLION", "0.25"))
        output_rate = float(os.environ.get("EMBODIED_RAG_OUTPUT_USD_PER_MILLION", "1.5"))
        estimated = (
            usage.prompt_tokens * input_rate + usage.completion_tokens * output_rate
        ) / 1_000_000.0
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": time.time(),
                "run_id": os.environ.get("INHERIT_SG_EXPERIMENT_RUN_ID", "embodied_rag_preflight"),
                "endpoint": "embodied_rag_text",
                "provider": "gateway",
                "model": kwargs.get("model"),
                "response_model": getattr(response, "model", None),
                "task": task,
                "cached": False,
                "latency_s": time.perf_counter() - started,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "estimated_cost_usd": estimated,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return response
