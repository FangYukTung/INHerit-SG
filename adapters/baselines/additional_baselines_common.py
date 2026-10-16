#!/usr/bin/env python3
"""Shared, dependency-light utilities for the additional baseline evaluation.

This module deliberately never reads ground-truth query fields.  Evaluation
adapters receive only the public split row (most importantly ``query_text``)
and method-produced maps.
"""

from __future__ import annotations

import hashlib
import fcntl
import http.client
import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = REPO_ROOT / "data/splits/main_864.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/reviewed"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/additional_baselines/evaluation_main864"
DEFAULT_SECRET_FILE = REPO_ROOT / ".secrets/baseline_api.env"


def load_env_file(path: Path = DEFAULT_SECRET_FILE) -> None:
    """Load a mode-600 KEY=VALUE file without logging any values."""
    path = Path(path)
    if not path.exists():
        return
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"Secret file must not be group/world accessible: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Malformed environment line in {path}")
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Multiple scene shards may resolve the same cached prompt concurrently.
    # A process-unique temporary prevents one writer from replacing another
    # writer's staging file; the final content is deterministic for the key.
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_split_rows(path: Path = DEFAULT_SPLIT) -> list[Dict[str, Any]]:
    data = load_json(path)
    rows = data.get("queries", [])
    if len(rows) != 864:
        raise ValueError(f"Expected Main-864, found {len(rows)} rows in {path}")
    keys = [row["key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate query keys in {path}")
    return rows


def rows_for_scene(rows: Iterable[Dict[str, Any]], scene_id: str) -> list[Dict[str, Any]]:
    return [row for row in rows if row["scene_id"] == scene_id]


def cache_key(model: str, system: str, prompt: str, schema_version: int = 1) -> str:
    payload = json.dumps(
        {"schema_version": schema_version, "model": model, "system": system, "prompt": prompt},
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response did not contain a JSON object")


# Relays behind Cloudflare (e.g. mulerouter) reject default library UAs with
# error 1010, so every request carries an explicit browser-like User-Agent.
REQUEST_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


class OpenAICompatibleClient:
    """Small OpenAI-compatible client with bounded retries and cached usage."""

    def __init__(self, cache_dir: Path, timeout_s: float = 120.0, max_attempts: int = 4):
        key = os.environ.get("YUNWU_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("YUNWU_API_KEY or OPENAI_API_KEY is not set")
        self.api_key = key
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(os.environ.get("BASELINE_API_TIMEOUT_S", timeout_s))
        self.max_attempts = int(os.environ.get("BASELINE_API_MAX_ATTEMPTS", max_attempts))

    def _budget_total(self) -> float:
        path_value = os.environ.get("BASELINE_USAGE_LOG")
        if not path_value:
            return 0.0
        path = Path(path_value)
        if not path.exists():
            return 0.0
        total = 0.0
        with path.open(encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            for line in handle:
                try:
                    total += float(json.loads(line).get("estimated_cost_usd", 0.0) or 0.0)
                except (json.JSONDecodeError, ValueError):
                    continue
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return total

    def _budget_preflight(self) -> None:
        limit_value = os.environ.get("BASELINE_MAX_COST_USD")
        if limit_value is not None and self._budget_total() >= float(limit_value):
            raise RuntimeError(
                f"Baseline API budget exhausted: ${self._budget_total():.6f}/${float(limit_value):.6f}"
            )

    def _record_usage(self, payload: Dict[str, Any], model: str, task: str) -> None:
        path_value = os.environ.get("BASELINE_USAGE_LOG")
        if not path_value:
            return
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        input_rate = float(os.environ.get("BASELINE_INPUT_USD_PER_MILLION", "0"))
        output_rate = float(os.environ.get("BASELINE_OUTPUT_USD_PER_MILLION", "0"))
        cost = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
        record = {
            "timestamp": time.time(), "model": model, "served_model": payload.get("model"),
            "task": task, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "estimated_cost_usd": cost,
        }
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            total = 0.0
            for line in handle:
                try:
                    total += float(json.loads(line).get("estimated_cost_usd", 0.0) or 0.0)
                except (json.JSONDecodeError, ValueError):
                    continue
            limit = float(os.environ.get("BASELINE_MAX_COST_USD", "inf"))
            if total + cost > limit:
                raise RuntimeError(
                    f"Baseline API budget exhausted: ${total + cost:.6f}/${limit:.6f}"
                )
            handle.seek(0, 2)
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int = 800,
        cache_namespace: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        digest = cache_key(model, system, prompt)
        cache_path = self.cache_dir / cache_namespace / f"{digest}.json"
        cached = load_json(cache_path)
        if cached is not None:
            return cached["parsed"], cached["metadata"]

        self._budget_preflight()

        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        # o-series models reject temperature and use max_completion_tokens.
        if model.lower().startswith("o"):
            body["max_completion_tokens"] = max_tokens
        else:
            body.update({"temperature": 0.0, "max_tokens": max_tokens})
        if model.lower().startswith("qwen"):
            # Match the frozen INHerit-SG text route.  The gateway otherwise
            # enables hidden reasoning by default, which changes both latency
            # and billed completion tokens for these short parser decisions.
            body["enable_thinking"] = False
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "User-Agent": REQUEST_USER_AGENT},
            method="POST",
        )
        last_error: Exception | None = None
        started = time.time()
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self._record_usage(payload, model, cache_namespace)
                content = payload["choices"][0]["message"]["content"]
                parsed = extract_json_object(content)
                metadata = {
                    "requested_model": model,
                    "served_model": payload.get("model"),
                    "usage": payload.get("usage", {}),
                    "latency_s": round(time.time() - started, 3),
                    "attempts": attempt,
                    "cache_key": digest,
                }
                atomic_json(cache_path, {"parsed": parsed, "metadata": metadata})
                return parsed, metadata
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    socket.timeout, ConnectionError, http.client.RemoteDisconnected,
                    KeyError, ValueError) as error:
                last_error = error
                if attempt == self.max_attempts:
                    break
                time.sleep(min(12.0, (2 ** (attempt - 1)) + random.random()))
        raise RuntimeError(f"OpenAI-compatible request failed after {self.max_attempts} attempts: {last_error}")

    def chat_text(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int = 800,
        cache_namespace: str,
    ) -> tuple[str, Dict[str, Any]]:
        """Call the native text interface and cache the unmodified response."""
        digest = cache_key(model, system, prompt)
        cache_path = self.cache_dir / cache_namespace / f"{digest}.json"
        cached = load_json(cache_path)
        if cached is not None:
            return cached["text"], cached["metadata"]

        self._budget_preflight()

        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if model.lower().startswith("o"):
            body["max_completion_tokens"] = max_tokens
        else:
            body.update({"temperature": 0.0, "max_tokens": max_tokens})
        if model.lower().startswith("qwen"):
            body["enable_thinking"] = False
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "User-Agent": REQUEST_USER_AGENT},
            method="POST",
        )
        last_error: Exception | None = None
        started = time.time()
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self._record_usage(payload, model, cache_namespace)
                content = str(payload["choices"][0]["message"]["content"]).strip()
                metadata = {
                    "requested_model": model,
                    "served_model": payload.get("model"),
                    "usage": payload.get("usage", {}),
                    "latency_s": round(time.time() - started, 3),
                    "attempts": attempt,
                    "cache_key": digest,
                }
                atomic_json(cache_path, {"text": content, "metadata": metadata})
                return content, metadata
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    socket.timeout, ConnectionError, http.client.RemoteDisconnected,
                    KeyError, ValueError) as error:
                last_error = error
                if attempt == self.max_attempts:
                    break
                time.sleep(min(12.0, (2 ** (attempt - 1)) + random.random()))
        raise RuntimeError(f"OpenAI-compatible request failed after {self.max_attempts} attempts: {last_error}")


def prediction_record(row: Dict[str, Any], method: str, candidates: list[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    """Create the exact schema consumed by aggregate_retrieval.py."""
    return {
        "schema_version": 1,
        "method": method,
        **{key: row[key] for key in ("key", "scene_id", "query_id", "query_type", "coarse_group")},
        "query_text": row["query_text"],
        "prediction": candidates[0] if candidates else None,
        "candidates": candidates[:5],
        "alignment": {"uses_gt_objects": False, "identity_threshold_m": 1.0},
        **extra,
    }
