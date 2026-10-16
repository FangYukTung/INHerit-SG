"""Narrow runtime compatibility fixes for the frozen KeySG release.

The upstream sampler asks sklearn's HDBSCAN for ``min_cluster_size=15``.  For
rooms with fewer than 15 assigned frames, sklearn raises before KeySG can take
its documented fallback to all room frames.  This module leaves the upstream
code untouched and restores that fallback only for this undefined case.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
from pathlib import Path


def install_clip_device_override() -> None:
    """Force KeySG's CLIP onto the device named by KEYSG_CLIP_DEVICE.

    The upstream default config hardcodes ``device="cuda"``.  Retrieval-only
    runs that share a build GPU with a map worker OOM, while a run with CUDA
    hidden fails to load the model at all.  Overriding the default config
    keeps the change local to this process.
    """
    device = os.environ.get("KEYSG_CLIP_DEVICE")
    if not device:
        return
    from keysg.utils import clip_utils

    clip_utils.DEFAULT_CLIP_CONFIG["device"] = device


def install_small_room_keyframe_fallback() -> None:
    """Make undersized rooms use KeySG's existing all-frames fallback."""
    from keysg.utils.frame_sampler import HDBSCANKeyframeSampler

    if getattr(HDBSCANKeyframeSampler, "_inherit_sg_small_room_patch", False):
        return

    original_sample_hdbscan = HDBSCANKeyframeSampler.sample_hdbscan

    def sample_hdbscan_with_small_room_fallback(
        self,
        min_cluster_size: int = 5,
        min_samples=None,
        rot_weight: float = 1.5,
        verbose: bool = True,
    ):
        if len(self.indices) < min_cluster_size:
            if verbose:
                print(
                    "HDBSCAN fallback: "
                    f"{len(self.indices)} frames < min_cluster_size={min_cluster_size}; "
                    "returning no medoids so KeySG retains all room frames."
                )
            return []
        return original_sample_hdbscan(
            self,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            rot_weight=rot_weight,
            verbose=verbose,
        )

    HDBSCANKeyframeSampler.sample_hdbscan = sample_hdbscan_with_small_room_fallback
    HDBSCANKeyframeSampler._inherit_sg_small_room_patch = True


def install_embedding_batch_limit(default_tokens: int = 50_000) -> None:
    """Split large embedding requests for relays that reject huge batches.

    The released client defaults to 300k aggregate tokens per request.  This
    compatibility layer only changes HTTP batching; it preserves the same
    input texts and resulting vectors and avoids repeating visual/API stages
    when a relay rejects one oversized embedding request.
    """
    from models.llm.openai_api import GPTInterface

    if getattr(GPTInterface, "_inherit_sg_embedding_batch_patch", False):
        return

    original_embed_text = GPTInterface.embed_text
    limit = int(os.environ.get("KEYSG_EMBED_MAX_TOTAL_TOKENS", default_tokens))
    max_chars = int(os.environ.get("KEYSG_EMBED_MAX_CHARS", "20000"))

    def embed_text_with_batch_limit(self, input_text, model="text-embedding-3-small", *args, **kwargs):
        caller_wants_usage = bool(kwargs.get("return_usage", False))
        usage_path = os.environ.get("KEYSG_USAGE_LOG")
        if usage_path:
            kwargs["return_usage"] = True
        kwargs.setdefault("max_total_tokens", limit)
        # The upstream graph builder truncates at 30k characters, which can
        # still exceed the relay/model's 8192-token per-input limit for JSON
        # or non-ASCII text.  Shorten only the request payload; keep the graph
        # chunks and their hashes unchanged for cache validation.
        if max_chars > 0:
            if isinstance(input_text, str):
                input_text = input_text[:max_chars]
            elif isinstance(input_text, (list, tuple)):
                input_text = [str(text)[:max_chars] for text in input_text]
        max_items = int(os.environ.get("KEYSG_EMBED_MAX_ITEMS", "0"))
        batch_cache = None
        cache_root = os.environ.get("KEYSG_EMBED_CACHE_DIR")
        if cache_root and max_items > 0 and isinstance(input_text, (list, tuple)) and len(input_text) <= max_items:
            import hashlib
            key = hashlib.sha256(json.dumps([model, input_text, args,
                {k: v for k, v in kwargs.items() if k != 'return_usage'}], sort_keys=True).encode()).hexdigest()
            batch_cache = Path(cache_root) / (key + '.json')
            if batch_cache.exists():
                cached = json.loads(batch_cache.read_text())
                assert len(cached['embeddings']) == len(input_text)
                return dict(cached, usage=[]) if caller_wants_usage else cached['embeddings']
        if max_items > 0 and isinstance(input_text, (list, tuple)) and len(input_text) > max_items:
            # Recovery-only HTTP batching. Preserve every input and its order;
            # do not aggregate vectors from different input texts.
            parts = []
            for start in range(0, len(input_text), max_items):
                # Account each completed batch immediately, even if a later
                # request fails; the enclosing guard holds its reservation.
                part = embed_text_with_batch_limit(self, input_text[start:start + max_items],
                                                   model, *args, **kwargs)
                parts.append(part)
                print(f"KeySG embedding batch complete: {min(start + max_items, len(input_text))}/{len(input_text)}", flush=True)
            if kwargs.get("return_usage"):
                result = {"embeddings": [], "usage": [], "model": None}
                for part in parts:
                    result["embeddings"].extend(part["embeddings"])
                    usage = part.get("usage")
                    result["usage"].extend(usage if isinstance(usage, list) else [usage])
                    result["model"] = part.get("model")
            else:
                result = [vector for part in parts for vector in part]
            return result if caller_wants_usage or not usage_path else result['embeddings']
        else:
            bounded = GPTInterface(self.client.with_options(
                timeout=float(os.environ.get("KEYSG_EMBED_TIMEOUT", "60")),
                max_retries=1,
            )) if max_items > 0 else self
            result = original_embed_text(bounded, input_text, model, *args, **kwargs)
        if usage_path and isinstance(result, dict):
            import fcntl
            import time

            usage_values = result.get("usage")
            usage_values = usage_values if isinstance(usage_values, list) else [usage_values]
            prompt_tokens = sum(
                int(getattr(usage, "prompt_tokens", 0) or 0)
                for usage in usage_values if usage is not None
            )
            rate = float(os.environ.get("KEYSG_EMBED_INPUT_USD_PER_MILLION", "0.02"))
            cost = prompt_tokens * rate / 1_000_000
            path = Path(usage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": time.time(), "model": model,
                "served_model": result.get("model"), "prompt_tokens": prompt_tokens,
                "completion_tokens": 0, "estimated_cost_usd": cost,
            }
            with path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.seek(0)
                total = 0.0
                for line in handle:
                    try:
                        total += float(json.loads(line).get("estimated_cost_usd", 0.0) or 0.0)
                    except (json.JSONDecodeError, ValueError):
                        continue
                limit_usd = float(os.environ.get("KEYSG_MAX_COST_USD", "inf"))
                if total + cost > limit_usd:
                    raise RuntimeError(
                        f"KeySG API budget exhausted: ${total + cost:.6f}/${limit_usd:.6f}"
                    )
                handle.seek(0, 2)
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if batch_cache is not None:
                batch_cache.parent.mkdir(parents=True, exist_ok=True)
                temp_cache = batch_cache.with_suffix(f'.{os.getpid()}.tmp')
                temp_cache.write_text(json.dumps({'embeddings': result['embeddings'], 'model': result.get('model')}))
                temp_cache.replace(batch_cache)
            if not caller_wants_usage:
                return result["embeddings"]
        return result

    GPTInterface.embed_text = embed_text_with_batch_limit
    GPTInterface._inherit_sg_embedding_batch_patch = True


def install_pose_only_room_assignment() -> None:
    """Avoid re-decoding RGB-D frames while assigning poses to one room."""
    from keysg.scene_segmentor.scene_segmentor import SceneSegmentor

    if getattr(SceneSegmentor, "_inherit_sg_pose_only_patch", False):
        return

    original_assign = SceneSegmentor._assign_poses_to_rooms

    def assign_with_pose_only_dataset(self, *args, **kwargs):
        dataset = self.dataset
        had_flag = hasattr(dataset, "_pose_only")
        previous = getattr(dataset, "_pose_only", False)
        # The real-robot adapter does not need to expose an otherwise-public
        # flag; set it only for the duration of this upstream pass.
        dataset._pose_only = True
        try:
            return original_assign(self, *args, **kwargs)
        finally:
            if had_flag:
                dataset._pose_only = previous
            else:
                delattr(dataset, "_pose_only")

    SceneSegmentor._assign_poses_to_rooms = assign_with_pose_only_dataset
    SceneSegmentor._inherit_sg_pose_only_patch = True


def install_pose_only_frame_sampler() -> None:
    """Avoid decoding RGB-D frames while HDBSCAN collects only poses."""
    from keysg.utils.frame_sampler import HDBSCANKeyframeSampler

    if getattr(HDBSCANKeyframeSampler, "_inherit_sg_pose_only_patch", False):
        return

    original_init = HDBSCANKeyframeSampler.__init__

    def init_with_pose_only(self, dataset, selected_indices=None):
        had_flag = hasattr(dataset, "_pose_only")
        previous = getattr(dataset, "_pose_only", False)
        dataset._pose_only = True
        try:
            return original_init(self, dataset, selected_indices=selected_indices)
        finally:
            if had_flag:
                dataset._pose_only = previous
            else:
                delattr(dataset, "_pose_only")

    HDBSCANKeyframeSampler.__init__ = init_with_pose_only
    HDBSCANKeyframeSampler._inherit_sg_pose_only_patch = True


def install_chat_completions_adapter(default_concurrency: int = 2) -> None:
    """Use Chat Completions for relays that do not implement Responses API.

    OpenLux exposes an OpenAI-compatible Chat Completions endpoint, while the
    upstream KeySG client currently calls ``responses.parse``.  This adapter is
    scoped to the runtime process and keeps the upstream checkout untouched.
    Pydantic schemas are sent as ``json_schema`` so structured VLM outputs keep
    the same validation contract as the original client.
    """
    from models.llm.openai_api import GPTInterface

    if getattr(GPTInterface, "_inherit_sg_chat_completions_patch", False):
        return

    def _selected_model(model, image):
        override = (
            os.environ.get("KEYSG_VISION_MODEL_OVERRIDE")
            if image is not None
            else os.environ.get("KEYSG_TEXT_MODEL_OVERRIDE")
        )
        return override or model

    def _record_usage(response, requested_model):
        usage_path = os.environ.get("KEYSG_USAGE_LOG")
        if not usage_path:
            return
        import fcntl
        import time

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if requested_model == "qwen3-vl-flash":
            input_rate, output_rate = 0.05, 0.4
        elif requested_model in {"qwen3.6-flash", "qwen3.6-35b-a3b"}:
            input_rate, output_rate = 0.25, 1.5
        elif requested_model == "gemini-2.5-flash":
            input_rate, output_rate = 0.30, 2.50
        else:
            input_rate = float(os.environ.get("KEYSG_INPUT_USD_PER_MILLION", "0"))
            output_rate = float(os.environ.get("KEYSG_OUTPUT_USD_PER_MILLION", "0"))
        cost = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
        path = Path(usage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.time(), "model": requested_model,
            "served_model": getattr(response, "model", None),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "estimated_cost_usd": cost,
        }
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            total = 0.0
            for line in handle:
                try:
                    total += float(json.loads(line).get("estimated_cost_usd", 0.0) or 0.0)
                except (json.JSONDecodeError, ValueError):
                    continue
            limit = float(os.environ.get("KEYSG_MAX_COST_USD", "inf"))
            if total + cost > limit:
                raise RuntimeError(
                    f"KeySG API budget exhausted: ${total + cost:.6f}/${limit:.6f}"
                )
            handle.seek(0, 2)
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _schema(response_model, tag_cap=None):
        raw = (
            response_model.model_json_schema()
            if hasattr(response_model, "model_json_schema")
            else response_model.schema()
        )
        definitions = raw.pop("$defs", {})

        def expand_refs(node):
            if isinstance(node, dict):
                ref = node.get("$ref")
                if ref and ref.startswith("#/$defs/"):
                    target = definitions.get(ref.rsplit("/", 1)[-1])
                    if target is not None:
                        return expand_refs(copy.deepcopy(target))
                return {key: expand_refs(value) for key, value in node.items() if key != "$defs"}
            if isinstance(node, list):
                return [expand_refs(value) for value in node]
            return node

        def make_strict(node):
            if isinstance(node, dict):
                if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                    node["additionalProperties"] = False
                    # OpenAI-compatible strict schemas require every declared
                    # property to be present; nullable/default fields retain
                    # their optional semantics through their JSON types.
                    node["required"] = list(node["properties"].keys())
                    # qwen3-vl-flash occasionally loops inside the free-text
                    # reason or the rejected_ids array until max_tokens cuts
                    # the JSON mid-string.  Bounding both keeps guided
                    # decoding from emitting unbounded runs.
                    props = node["properties"]
                    # Enforce the tagging request's existing cap in the schema:
                    # Qwen otherwise sometimes repeats tags until truncation.
                    tags = props.get("tags")
                    if tag_cap is not None and isinstance(tags, dict) and tags.get("type") == "array":
                        tags.setdefault("maxItems", tag_cap)
                    reason = props.get("reason")
                    if isinstance(reason, dict) and reason.get("type") == "string":
                        reason.setdefault("maxLength", 600)
                    rejected = props.get("rejected_ids")
                    if isinstance(rejected, dict) and rejected.get("type") == "array":
                        rejected.setdefault("maxItems", 32)
                for value in node.values():
                    make_strict(value)
            elif isinstance(node, list):
                for value in node:
                    make_strict(value)
            return node

        return make_strict(expand_refs(raw))

    def _validate(response_model, text):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            # Keep a sample of the malformed payload in the error message so
            # the cached error record shows what the relay actually returned.
            raise json.JSONDecodeError(
                f"{error.msg} | response_head={text[:200]!r} response_tail={text[-200:]!r}",
                error.doc,
                error.pos,
            ) from error
        validate = (
            response_model.model_validate
            if hasattr(response_model, "model_validate")
            else response_model.parse_obj
        )
        try:
            return validate(payload)
        except Exception:
            # qwen models sometimes nest the answer in a single wrapper key
            # ({"scene_summary": {...}}) despite the strict schema; unwrap it.
            if isinstance(payload, dict) and len(payload) == 1:
                inner = next(iter(payload.values()))
                if isinstance(inner, dict):
                    return validate(inner)
            raise

    def _content_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    def _messages(prompt, instructions, image, detail, encoder):
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})

        if isinstance(prompt, str):
            user_content = [{"type": "text", "text": prompt}]
        else:
            # Current KeySG calls use string prompts, but preserve the common
            # OpenAI message-list form for future structured calls.
            for msg in prompt:
                if msg.get("role") == "system" and msg.get("content"):
                    messages.append({"role": "system", "content": msg["content"]})
            user_content = []
            for msg in prompt:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_content.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        user_content.extend(content)

        if image is not None:
            images = image if isinstance(image, list) else [image]
            for img in images:
                url = img if isinstance(img, str) else encoder(img)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url, "detail": detail},
                    }
                )
        messages.append({"role": "user", "content": user_content})
        return messages

    def chat_structured(
        self,
        prompt,
        response_model,
        model="gpt-5-mini",
        stream=False,
        image=None,
        detail="auto",
        **kwargs,
    ):
        if stream:
            raise NotImplementedError("Streaming structured output is not used by KeySG")
        model = _selected_model(model, image)
        messages = _messages(
            prompt, kwargs.pop("instructions", None), image, detail, self._encode_image
        )
        # DashScope-compatible relays require an explicit JSON-format cue when
        # response_format is enabled, even when a full schema is supplied.
        messages.insert(0, {"role": "system", "content": "Return valid JSON only."})
        # The relay may reset one oversized/concurrent request.  Disable the
        # SDK's default multi-minute retry loop so a failed frame is isolated.
        client = self.client.with_options(
            timeout=float(os.environ.get("KEYSG_VLM_TIMEOUT", "90")),
            max_retries=0,
        )
        extra_body = {"enable_thinking": False} if model.lower().startswith("qwen") else None
        from api_usage import guarded_chat
        attempts = max(1, int(os.environ.get("KEYSG_API_MAX_ATTEMPTS", "3")))
        extra = {}
        for attempt in range(1, attempts + 1):
            response = None
            try:
                response = guarded_chat(client.chat.completions.create, _record_usage,
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_model.__name__,
                            "strict": True,
                            "schema": _schema(response_model, tag_cap=(
                                int(match.group(1)) if (match := re.search(r"Cap at (\d+) items", str(prompt))) else None
                            )),
                        },
                    },
                    temperature=0,
                    extra_body=extra_body,
                    # Only retries of a malformed response get an anti-loop
                    # penalty; first attempts keep the frozen request shape.
                    **extra,
                )
                return _validate(response_model, _content_text(response.choices[0].message.content))
            except Exception as error:
                if attempt == attempts:
                    raise
                if isinstance(error, json.JSONDecodeError):
                    extra = {"frequency_penalty": 0.5}
                    # Gemini's completion budget includes reasoning. A rare
                    # long-thinking selection can reach the limit before its
                    # short JSON finishes. Retry the identical task with more
                    # output headroom; never repair/truncate predictions by hand.
                    if (model == "gemini-2.5-flash" and response is not None
                            and response.choices[0].finish_reason == "length"):
                        extra["max_tokens"] = 16384
                print(f"KeySG structured request retry {attempt}/{attempts - 1}: {type(error).__name__}", flush=True)
                time.sleep(min(8, 2 ** (attempt - 1)))

    async def chat_structured_one(
        self,
        prompt,
        response_model,
        model,
        image,
        detail,
        semaphore,
        **kwargs,
    ):
        async with semaphore:
            return await asyncio.to_thread(
                chat_structured,
                self,
                prompt,
                response_model,
                model,
                False,
                image,
                detail,
                **kwargs,
            )

    async def chat_structured_batch(
        self,
        prompts,
        response_model,
        model="gpt-5-mini",
        images=None,
        detail="auto",
        **kwargs,
    ):
        if images is not None and len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of image entries.")
        limit = max(1, int(os.environ.get("KEYSG_VLM_CONCURRENCY", default_concurrency)))
        semaphore = asyncio.Semaphore(limit)
        tasks = [
            chat_structured_one(
                self,
                prompt,
                response_model,
                model,
                images[i] if images is not None else None,
                detail,
                semaphore,
                **kwargs,
            )
            for i, prompt in enumerate(prompts)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def chat_text(self, prompt, model="gpt-5-mini", stream=False, **kwargs):
        if stream:
            raise NotImplementedError("Streaming text output is not used by KeySG")
        image = kwargs.pop("image", None)
        model = _selected_model(model, image)
        messages = _messages(
            prompt, kwargs.pop("instructions", None), image,
            kwargs.pop("detail", "auto"), self._encode_image
        )
        extra_body = {"enable_thinking": False} if model.lower().startswith("qwen") else None
        from api_usage import guarded_chat
        attempts = max(1, int(os.environ.get("KEYSG_API_MAX_ATTEMPTS", "3")))
        for attempt in range(1, attempts + 1):
            try:
                response = guarded_chat(self.client.with_options(timeout=90, max_retries=0).chat.completions.create, _record_usage,
                    model=model, messages=messages, temperature=0, extra_body=extra_body
                )
                return _content_text(response.choices[0].message.content)
            except Exception as error:
                if attempt == attempts:
                    raise
                print(f"KeySG text request retry {attempt}/{attempts - 1}: {type(error).__name__}", flush=True)
                time.sleep(min(8, 2 ** (attempt - 1)))

    def chat_vision(self, prompt, image, model="gpt-5-mini", stream=False, **kwargs):
        kwargs["image"] = image
        return chat_text(self, prompt, model=model, stream=stream, **kwargs)

    GPTInterface.structured_prompt = chat_structured
    GPTInterface._process_one_structured_prompt = chat_structured_one
    GPTInterface.structured_prompt_batch = chat_structured_batch
    GPTInterface.text_prompt = chat_text
    GPTInterface.vision_prompt = chat_vision
    GPTInterface._inherit_sg_chat_completions_patch = True
