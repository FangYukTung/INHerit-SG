import hashlib
import fcntl
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from openai import OpenAI


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _response_dict(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    raise TypeError(f"Unsupported response type: {type(response)!r}")


class _InstrumentedCompletions:
    def __init__(self, endpoint: "InstrumentedEndpoint", task: Optional[str] = None):
        self.endpoint = endpoint
        self.task = task

    def create(self, **kwargs):
        kwargs["_router_task"] = self.task
        return self.endpoint.create_completion(**kwargs)


class _InstrumentedChat:
    def __init__(self, endpoint: "InstrumentedEndpoint", task: Optional[str] = None):
        self.completions = _InstrumentedCompletions(endpoint, task)


class _TaskBoundEndpoint:
    def __init__(self, endpoint: "InstrumentedEndpoint", task: str):
        self.chat = _InstrumentedChat(endpoint, task)
        self.base_url = endpoint.base_url


class InstrumentedEndpoint:
    """OpenAI-compatible client with deterministic disk cache and usage logs."""

    _write_lock = threading.Lock()

    def __init__(
        self,
        name: str,
        settings: Dict[str, Any],
        cache_dir: Optional[str] = None,
        usage_log: Optional[str] = None,
        max_cost_usd: Optional[float] = None,
        max_api_calls: Optional[int] = None,
    ):
        self.name = name
        self.settings = settings
        api_key = os.environ.get(settings.get("api_key_env", ""), settings.get("api_key", ""))
        if not api_key:
            raise ValueError(
                f"Missing API key for endpoint '{name}'. Set {settings.get('api_key_env', 'the configured variable')}."
            )
        base_url = os.environ.get(settings.get("base_url_env", "")) or settings.get("base_url")
        if settings.get("base_url_env") and not base_url:
            raise ValueError(f"Set {settings['base_url_env']} for endpoint '{name}'")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(settings.get("timeout_s", 60.0)),
            max_retries=int(settings.get("max_retries", 2)),
        )
        self.base_url = self.client.base_url
        self.chat = _InstrumentedChat(self)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.usage_log = Path(usage_log).expanduser() if usage_log else None
        env_cost = os.environ.get("INHERIT_SG_MAX_COST_USD")
        env_calls = os.environ.get("INHERIT_SG_MAX_API_CALLS")
        self.max_cost_usd = float(max_cost_usd if max_cost_usd is not None else env_cost) if (max_cost_usd is not None or env_cost) else None
        self.max_api_calls = int(max_api_calls if max_api_calls is not None else env_calls) if (max_api_calls is not None or env_calls) else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.usage_log:
            self.usage_log.parent.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, kwargs: Dict[str, Any]) -> str:
        payload = json.dumps(
            {"endpoint": self.name, "request": kwargs},
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _estimate_cost(self, response: Dict[str, Any]) -> float:
        usage = response.get("usage") or {}
        prompt = usage.get("prompt_tokens", 0) or 0
        completion = usage.get("completion_tokens", 0) or 0
        input_rate = float(self.settings.get("input_usd_per_million", 0.0))
        output_rate = float(self.settings.get("output_usd_per_million", 0.0))
        return (prompt * input_rate + completion * output_rate) / 1_000_000.0

    def _append_log(self, record: Dict[str, Any]) -> None:
        if not self.usage_log:
            return
        with self._write_lock:
            with self.usage_log.open("a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _budget_usage(self):
        if not self.usage_log or not self.usage_log.exists():
            return 0.0, 0
        cost = 0.0
        calls = 0
        with self.usage_log.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record.get("cached", False):
                    calls += 1
                    cost += float(record.get("estimated_cost_usd", 0.0) or 0.0)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return cost, calls

    def _check_budget(self):
        cost, calls = self._budget_usage()
        if self.max_cost_usd is not None and cost >= self.max_cost_usd:
            raise RuntimeError(
                f"API cost budget exhausted: ${cost:.6f} >= ${self.max_cost_usd:.6f}"
            )
        if self.max_api_calls is not None and calls >= self.max_api_calls:
            raise RuntimeError(
                f"API call budget exhausted: {calls} >= {self.max_api_calls}"
            )

    def create_completion(self, **kwargs):
        task = kwargs.pop("_router_task", None)
        cache_salt = kwargs.pop("_router_cache_salt", None)
        if not self.settings.get("supports_vision", False) and self._contains_image(kwargs.get("messages", [])):
            raise ValueError(
                f"Task '{task or 'unknown'}' sent an image to text-only endpoint "
                f"'{self.name}' ({self.settings.get('model', 'unknown model')})."
            )
        if self.settings.get("extra_body"):
            kwargs.setdefault("extra_body", self.settings["extra_body"])
        cache_request = dict(kwargs)
        if cache_salt is not None:
            cache_request["_router_cache_salt"] = cache_salt
        cache_key = self._cache_key(cache_request)
        cache_path = self.cache_dir / f"{cache_key}.json" if self.cache_dir else None
        started = time.perf_counter()
        cached = bool(cache_path and cache_path.exists())

        if cached:
            with cache_path.open("r", encoding="utf-8") as handle:
                response_dict = json.load(handle)
            response = _namespace(response_dict)
        else:
            self._check_budget()
            response = self.client.chat.completions.create(**kwargs)
            response_dict = _response_dict(response)
            if cache_path:
                temporary = cache_path.with_suffix(".tmp")
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(response_dict, handle, ensure_ascii=True)
                os.replace(temporary, cache_path)

        elapsed = time.perf_counter() - started
        usage = response_dict.get("usage") or {}
        self._append_log(
            {
                "timestamp": time.time(),
                "run_id": os.environ.get("INHERIT_SG_EXPERIMENT_RUN_ID", "default"),
                "endpoint": self.name,
                "provider": self.settings.get("provider", "openai_compatible"),
                "model": kwargs.get("model"),
                "task": task,
                "cached": cached,
                "latency_s": elapsed,
                "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
                "total_tokens": usage.get("total_tokens", 0) or 0,
                "estimated_cost_usd": 0.0 if cached else self._estimate_cost(response_dict),
                "cache_key": cache_key,
            }
        )
        return response

    @staticmethod
    def _contains_image(messages):
        for message in messages or []:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"image_url", "input_image", "image"}:
                    return True
        return False

    def for_task(self, task: str):
        return _TaskBoundEndpoint(self, task)


class ModelRouter:
    def __init__(self, config: Dict[str, Any]):
        runtime = config.get("runtime", {})
        cache_dir = runtime.get("cache_dir")
        usage_log = runtime.get("usage_log")
        self.routes = config.get("routing", {})
        self.endpoints: Dict[str, InstrumentedEndpoint] = {}
        self.models: Dict[str, str] = {}
        referenced = {name for name in self.routes.values() if name}
        configured_endpoints = config.get("endpoints", {})
        unknown = referenced - set(configured_endpoints)
        if unknown:
            raise ValueError(f"Routes reference undefined endpoints: {sorted(unknown)}")
        for name, settings in configured_endpoints.items():
            if referenced and name not in referenced:
                continue
            self.endpoints[name] = InstrumentedEndpoint(
                name,
                settings,
                cache_dir,
                usage_log,
                runtime.get("max_cost_usd"),
                runtime.get("max_api_calls"),
            )
            self.models[name] = settings["model"]
        if not self.endpoints:
            raise ValueError("model_router.endpoints must define at least one endpoint")

    def resolve(self, task: str):
        endpoint_name = self.routes.get(task, self.routes.get("default"))
        if not endpoint_name:
            raise ValueError(f"No route or default endpoint configured for task '{task}'")
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Task '{task}' references unavailable endpoint '{endpoint_name}'")
        endpoint = self.endpoints[endpoint_name]
        return endpoint.for_task(task), self.models[endpoint_name]
