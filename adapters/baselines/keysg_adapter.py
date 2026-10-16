#!/usr/bin/env python3
"""Adapt an unmodified KeySG map to the HM3DSem-SQR prediction schema.

This module deliberately lives outside ``baselines/upstream-KeySG``.  It only
serializes the released KeySG retrieval results; benchmark GT is never read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import copy
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import openai


ROOT = Path(__file__).resolve().parents[2]


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("queries", payload)
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a list or contain a 'queries' list")
    return rows


def node_center(node: Any) -> list[float] | None:
    """Return a method-local node center without consulting simulator metadata."""
    pcd = getattr(node, "pcd", None)
    if pcd is not None:
        try:
            center = np.asarray(pcd.get_center(), dtype=float)
            if center.shape == (3,) and np.all(np.isfinite(center)):
                return center.tolist()
        except Exception:
            pass
    bbox = getattr(node, "bbox_3d", None)
    if bbox is not None:
        try:
            corners = np.asarray(bbox, dtype=float)
            if corners.shape == (8, 3) and np.all(np.isfinite(corners)):
                return corners.mean(axis=0).tolist()
        except Exception:
            pass
    return None


def candidate_record(node: Any, score: float, source: str) -> dict[str, Any] | None:
    position = node_center(node)
    if position is None:
        return None
    node_id = str(getattr(node, "id", "") or "")
    if not node_id:
        return None
    return {
        "id": node_id,
        "label": str(getattr(node, "label", "") or ""),
        "position": position,
        "score": float(score),
        "source": source,
    }


def ordered_candidates(
    selected_id: str | None,
    raw_results: Iterable[Any],
    objects_by_id: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Use released KeySG selection for rank 1 and native RAG order thereafter.

    KeySG's current public query script emits only an LLM-selected Top-1 even
    though the underlying object-visual retriever returns an ordered candidate
    list.  This is an interface conversion, not a reranker: after the released
    Top-1, candidates retain their source order and scores.
    """
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    if selected_id:
        selected = objects_by_id.get(str(selected_id))
        if selected is not None:
            record = candidate_record(selected, 1.0, "keysg_llm_selection")
            if record is not None:
                records.append(record)
                seen.add(record["id"])

    for result in raw_results:
        chunk = getattr(result, "chunk", None)
        object_id = str(getattr(chunk, "id", "") or "")
        if not object_id or object_id in seen:
            continue
        node = objects_by_id.get(object_id)
        if node is None:
            continue
        record = candidate_record(node, float(getattr(result, "score", 0.0)), "keysg_object_visual_rag")
        if record is None:
            continue
        records.append(record)
        seen.add(record["id"])
        if len(records) >= top_k:
            break
    return records


def _imports() -> tuple[Any, Any, Any, Any, Any, Any]:
    root = os.environ.get("KEYSG_ROOT")
    if not root:
        raise EnvironmentError("Set KEYSG_ROOT to the upstream KeySG checkout")
    upstream = Path(root).resolve()
    if not upstream.is_dir():
        raise FileNotFoundError(f"KeySG upstream checkout is missing: {upstream}")
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "scripts"))
    from keysg.rag.graph_context_retriever import GraphContextRetriever
    from keysg.rag.query_analysis import _QuerySchema, SYSTEM_INSTRUCTIONS
    from keysg.utils.load_utils import get_objects, load_scene_nodes
    from models.llm.openai_api import GPTInterface
    from scripts.nr3d_eval import ObjectSelection, _OBJECT_SELECTION_SYSTEM_PROMPT
    # Reuse these helpers rather than reproducing KeySG's score normalization
    # or its near/far geometry context in the benchmark adapter.
    from eval_helpers import (
        _build_spatial_relations,
        _compute_scene_center,
        _rank_frame_ids,
        _run_structured_batch,
    )
    return (
        GraphContextRetriever,
        _QuerySchema,
        SYSTEM_INSTRUCTIONS,
        get_objects,
        load_scene_nodes,
        (
            GPTInterface,
            ObjectSelection,
            _OBJECT_SELECTION_SYSTEM_PROMPT,
            _run_structured_batch,
            _rank_frame_ids,
            _compute_scene_center,
            _build_spatial_relations,
        ),
    )


def _install_runtime_compat() -> None:
    """Install the isolated relay compatibility layer for retrieval runs.

    The mapping launcher installs this patch before importing KeySG.  The
    retrieval adapter is a separate process, so it must do the same here;
    otherwise the released client calls the unsupported Responses API.
    """
    if os.environ.get("OPENLUX_KEY") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENLUX_KEY"]
    patch_root = Path(__file__).resolve().parent
    if str(patch_root) not in sys.path:
        sys.path.insert(0, str(patch_root))
    from keysg_runtime_patches import (
        install_chat_completions_adapter,
        install_clip_device_override,
        install_embedding_batch_limit,
    )

    install_embedding_batch_limit()
    install_chat_completions_adapter()
    install_clip_device_override()


def _load_frame_images(chunks: list[Any], max_images: int) -> list[Any]:
    from PIL import Image

    images = []
    for chunk in chunks:
        image_path = chunk.metadata.get("labeled_image_path") or chunk.metadata.get("image_path")
        if not image_path:
            continue
        path = Path(image_path)
        if not path.is_file():
            continue
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
        if len(images) >= max_images:
            break
    return images


def _safe_bbox(raw: Any, width: int, height: int) -> list[int] | None:
    try:
        values = [float(v) for v in raw]
    except Exception:
        return None
    if len(values) != 4:
        return None
    x0, y0, x1, y1 = values
    x0, x1 = sorted((max(0.0, min(width - 1.0, x0)), max(0.0, min(width - 1.0, x1))))
    y0, y1 = sorted((max(0.0, min(height - 1.0, y0)), max(0.0, min(height - 1.0, y1))))
    result = [round(x0), round(y0), round(x1), round(y1)]
    return result if result[2] > result[0] and result[3] > result[1] else None


def _write_node_evidence(node: Any, query_id: str, evidence_dir: Path) -> dict[str, Any] | None:
    """Serialize the best native KeySG view as full-frame + context crop.

    ObjNode keeps ``rgb_frames``, ``bboxs_2d`` and ``frame_indices`` together,
    so this does not consult simulator GT or infer a shared object ID.
    """
    from PIL import Image, ImageDraw

    frames = list(getattr(node, "rgb_frames", None) or [])
    boxes = list(getattr(node, "bboxs_2d", None) or [])
    frame_ids = list(getattr(node, "frame_indices", None) or [])
    if not frames or not boxes:
        return None
    # Prefer a frame with both a valid RGB image and a valid 2D box.  The
    # mapping code stores these arrays in aligned order; tolerate incomplete
    # entries from old checkpoints.
    chosen = None
    for index, (raw_image, raw_box) in enumerate(zip(frames, boxes)):
        try:
            image = Image.fromarray(np.asarray(raw_image).astype(np.uint8)).convert("RGB")
        except Exception:
            continue
        bbox = _safe_bbox(raw_box, image.width, image.height)
        if bbox is not None:
            frame_id = frame_ids[index] if index < len(frame_ids) else None
            chosen = (image, bbox, frame_id)
            break
    if chosen is None:
        return None
    image, bbox, frame_id = chosen
    target = evidence_dir / f"{query_id}.jpg"
    context_target = evidence_dir / f"{query_id}_context.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = image.copy()
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    width = max(3, round(max(image.size) / 300))
    draw.rectangle(tuple(bbox), outline=(255, 0, 0), width=width)
    boxed.save(target, quality=92)

    object_width = bbox[2] - bbox[0]
    object_height = bbox[3] - bbox[1]
    crop_width = min(image.width, max(512, object_width * 5))
    crop_height = min(image.height, max(512, object_height * 5))
    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    crop_x0 = max(0, min(image.width - crop_width, cx - crop_width // 2))
    crop_y0 = max(0, min(image.height - crop_height, cy - crop_height // 2))
    crop_box = (crop_x0, crop_y0, crop_x0 + crop_width, crop_y0 + crop_height)
    context = clean.crop(crop_box)
    context_bbox = [bbox[0] - crop_x0, bbox[1] - crop_y0, bbox[2] - crop_x0, bbox[3] - crop_y0]
    ImageDraw.Draw(context).rectangle(tuple(context_bbox), outline=(255, 0, 0), width=width)
    context.save(context_target, quality=92)
    return {
        "evidence_path": str(target.resolve()),
        "bbox_xyxy": bbox,
        "frame_id": int(frame_id) if frame_id is not None else None,
        "views": [
            {
                "image_path": str(context_target.resolve()),
                "role": "target_context_crop",
                "bbox_xyxy": context_bbox,
                "frame_id": int(frame_id) if frame_id is not None else None,
            },
            {
                "image_path": str(target.resolve()),
                "role": "full_scene_context",
                "bbox_xyxy": bbox,
                "frame_id": int(frame_id) if frame_id is not None else None,
            },
        ],
    }


def _run_batch_with_explicit_retries(
    gpt: Any,
    run_structured_batch: Any,
    prompts: list[str],
    images_list: list[Any] | None,
    *,
    response_model: Any,
    model: str,
    instructions: str,
    reasoning_effort: str,
    detail: str,
) -> list[Any]:
    """Retry only failed structured requests, never successful items.

    The routed endpoint occasionally returns a transient connection error for
    a single request in a concurrent batch.  Retrying the entire scene would
    duplicate successful API calls, so retries are explicitly scoped to the
    failed indices and are disabled by default for reproducibility.
    """
    results = run_structured_batch(
        gpt,
        prompts,
        images_list=images_list,
        response_model=response_model,
        model=model,
        instructions=instructions,
        reasoning_effort=reasoning_effort,
        detail=detail,
    )
    retry_count = max(0, int(os.getenv("KEYSG_ITEM_RETRIES", "0")))
    pending = [i for i, result in enumerate(results) if isinstance(result, Exception)]
    for _ in range(retry_count):
        if not pending:
            break
        retry_prompts = [prompts[i] for i in pending]
        retry_images = [images_list[i] for i in pending] if images_list is not None else None
        retry_results = run_structured_batch(
            gpt,
            retry_prompts,
            images_list=retry_images,
            response_model=response_model,
            model=model,
            instructions=instructions,
            reasoning_effort=reasoning_effort,
            detail=detail,
        )
        for index, result in zip(pending, retry_results):
            results[index] = result
        pending = [i for i, result in enumerate(results) if isinstance(result, Exception)]
    return results


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _resumable_structured_batch(
    gpt: Any,
    run_structured_batch: Any,
    prompts: list[str],
    images_list: list[Any] | None,
    *,
    response_model: Any,
    model: str,
    instructions: str,
    reasoning_effort: str,
    detail: str,
    cache_path: Path,
    batch_size: int,
) -> list[Any]:
    """Run structured requests in small durable batches.

    Cache keys include the full prompt and model.  A completed request is
    never sent again when a later batch or process is interrupted.
    """
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    except Exception:
        cache = {}
    if not isinstance(cache, dict):
        cache = {}
    outputs: list[Any] = [None] * len(prompts)
    pending: list[int] = []
    prefix = f"{response_model.__name__}\0{model}\0{instructions}\0{reasoning_effort}\0{detail}\0"
    for index, prompt in enumerate(prompts):
        key = hashlib.sha256((prefix + prompt).encode("utf-8")).hexdigest()
        entry = cache.get(key)
        if isinstance(entry, dict) and entry.get("status") == "ok":
            try:
                outputs[index] = response_model.model_validate(entry["value"])
            except AttributeError:
                outputs[index] = response_model.parse_obj(entry["value"])
        elif isinstance(entry, dict) and entry.get("status") == "error":
            outputs[index] = RuntimeError(entry.get("error", "cached request error"))
        else:
            pending.append(index)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(pending), max(1, batch_size)):
        indices = pending[offset : offset + max(1, batch_size)]
        batch_prompts = [prompts[i] for i in indices]
        batch_images = [images_list[i] for i in indices] if images_list is not None else None
        batch_results = run_structured_batch(
            gpt,
            batch_prompts,
            images_list=batch_images,
            response_model=response_model,
            model=model,
            instructions=instructions,
            reasoning_effort=reasoning_effort,
            detail=detail,
        )
        for index, prompt, value in zip(indices, batch_prompts, batch_results):
            key = hashlib.sha256((prefix + prompt).encode("utf-8")).hexdigest()
            if isinstance(value, Exception):
                outputs[index] = value
                cache[key] = {"status": "error", "error": f"{type(value).__name__}: {value}"}
            else:
                outputs[index] = value
                cache[key] = {"status": "ok", "value": _dump_model(value)}
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs


def _install_query_embedding_cache(
    retriever: Any,
    gpt: Any,
    queries: list[str],
    *,
    model: str,
    cache_path: Path,
) -> int:
    """Batch all per-query text embeddings once and serve them locally."""
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    except Exception:
        cached = {}
    if not isinstance(cached, dict):
        cached = {}
    unique = list(dict.fromkeys(str(q) for q in queries if str(q).strip()))
    missing = [q for q in unique if f"{model}\0{q}" not in cached]
    if missing:
        vectors = gpt.embed_text(missing, model=model)
        for query, vector in zip(missing, vectors):
            cached[f"{model}\0{query}"] = vector
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    vector_cache = {q: cached[f"{model}\0{q}"] for q in unique if f"{model}\0{q}" in cached}
    original = gpt.embed_text

    def embed_cached(input_text, embed_model=model, *args, **kwargs):
        items = [input_text] if isinstance(input_text, str) else list(input_text)
        if embed_model == model and all(str(item) in vector_cache for item in items):
            result = [vector_cache[str(item)] for item in items]
            return result[0] if isinstance(input_text, str) else result
        return original(input_text, embed_model, *args, **kwargs)

    gpt.embed_text = embed_cached
    # GraphContextRetriever keeps the same GPTInterface instance, but retain
    # this assignment for defensive compatibility with future lazy loaders.
    retriever.gpt = gpt
    return len(missing)


def _context(
    query: str,
    parsed: Any,
    target_results: list[Any],
    anchor_results: list[Any],
    frame_chunks: list[Any],
    spatial_relations: list[str],
) -> str:
    target = parsed.target_object or query
    anchors = list(parsed.anchor_objects or [])
    sections = [
        f"USER QUERY: {query}",
        f"PARSED TARGET: {target}",
        f"PARSED ANCHORS: {anchors}",
    ]
    if target_results:
        sections.append("Target Object Candidates:\n" + "\n".join(
            f"{index}. ID={result.chunk.id}. Desc={result.chunk.content}"
            for index, result in enumerate(target_results, start=1)
        ))
    if anchor_results:
        sections.append("Anchor Object Candidates:\n" + "\n".join(
            f"{index}. ID={result.chunk.id}. Desc={result.chunk.content}"
            for index, result in enumerate(anchor_results, start=1)
        ))
    if frame_chunks:
        sections.append("Relevant Frames:\n" + "\n".join(
            f"{index}. FRAME_ID={chunk.id}. {chunk.content}"
            for index, chunk in enumerate(frame_chunks, start=1)
        ))
    if spatial_relations:
        sections.append("Spatial Relations (target <-> anchor):\n" + "\n".join(spatial_relations))
    return "\n\n".join(sections)


def run(args: argparse.Namespace) -> None:
    _install_runtime_compat()
    (
        GraphContextRetriever,
        QuerySchema,
        query_instructions,
        get_objects,
        load_scene_nodes,
        query_support,
    ) = _imports()
    (
        GPTInterface,
        ObjectSelection,
        selection_instructions,
        run_structured_batch,
        rank_frame_ids,
        compute_scene_center,
        build_spatial_relations,
    ) = query_support

    rows = [row for row in load_rows(args.manifest) if row["scene_id"] == args.scene_id]
    if not rows:
        raise RuntimeError(f"No manifest rows for scene {args.scene_id}")
    scene_dir = Path(args.scene_dir)
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)

    retriever = GraphContextRetriever(str(scene_dir))
    retriever.build_chunks()
    retriever.compute_embeddings(
        model_name=args.embedding_model,
        compute_frame_visual=True,
        compute_object_visual=True,
        use_cache=True,
    )
    retriever.build_faiss_index(use_cache=True)
    objects = get_objects(load_scene_nodes(str(scene_dir)))
    objects_by_id = {str(object_.id): object_ for object_ in objects}
    scene_center = compute_scene_center(objects)
    chunk_by_id = {chunk.id: chunk for chunk in retriever.chunks}
    # The released helper otherwise inherits the OpenAI SDK's long default
    # timeout and automatic retries.  A transient routed-endpoint failure
    # could therefore repeat a whole 24-query batch and consume tokens again.
    # Keep requests bounded and opt out of hidden retries; the scene launcher
    # records a failed scene for an explicit, operator-controlled resume.
    gpt = GPTInterface(
        client=openai.OpenAI(
            timeout=float(os.getenv("KEYSG_API_TIMEOUT", "240")),
            max_retries=int(os.getenv("KEYSG_API_MAX_RETRIES", "0")),
        )
    )

    queries = [row["query_text"] for row in rows]
    cache_prefix = Path(args.output).with_suffix("")
    request_batch_size = max(1, int(os.environ.get("KEYSG_REQUEST_BATCH_SIZE", str(args.request_batch_size))))
    parsed = _resumable_structured_batch(
        gpt,
        run_structured_batch,
        [f"User query: {query}" for query in queries],
        images_list=None,
        response_model=QuerySchema,
        model=args.parser_model,
        instructions=query_instructions,
        reasoning_effort="low",
        detail="auto",
        cache_path=cache_prefix.with_name(cache_prefix.name + "_parser_cache.json"),
        batch_size=request_batch_size,
    )

    # Searches below call text embedding for each target/anchor/frame query.
    # Replace that N-request pattern with one batch and a durable scene cache.
    embedding_inputs: list[str] = []
    for query, parsed_query in zip(queries, parsed):
        if isinstance(parsed_query, Exception):
            target = query
            anchors: list[str] = []
        else:
            target = parsed_query.target_object or query
            anchors = list(parsed_query.anchor_objects or [])
        embedding_inputs.extend([target, query])
        if anchors:
            embedding_inputs.append(" ".join(anchors))
    missing_embeddings = _install_query_embedding_cache(
        retriever,
        gpt,
        embedding_inputs,
        model=args.embedding_model,
        cache_path=cache_prefix.with_name(cache_prefix.name + "_query_embedding_cache.json"),
    )
    if missing_embeddings:
        print(json.dumps({"scene_id": args.scene_id, "new_query_embeddings": missing_embeddings}))

    prepared: list[dict[str, Any]] = []
    for row, query, parsed_query in zip(rows, queries, parsed):
        parse_error = parsed_query if isinstance(parsed_query, Exception) else None
        if parse_error:
            target = query
            anchors: list[str] = []
            parsed_query = None
        else:
            target = parsed_query.target_object or query
            anchors = list(parsed_query.anchor_objects or [])
        target_results = retriever.search(
            target,
            top_k=args.raw_candidate_k,
            doc_types=["object"],
            object_modality="both",
        ).get("object_visual", [])
        anchor_results: list[Any] = []
        if anchors:
            anchor_results = retriever.search(
                " ".join(anchors),
                top_k=args.raw_candidate_k,
                doc_types=["object"],
                object_modality="both",
            ).get("object_visual", [])
        frame_results = retriever.search(
            query,
            top_k=args.frame_candidate_k,
            doc_types=["frame"],
            object_modality="both",
            frame_modality="both",
        )
        frame_chunks = [
            chunk_by_id[item]
            for item in rank_frame_ids(
                frame_results,
                args.frame_candidate_k,
                include_text=False,
                include_visual=True,
            )
            if item in chunk_by_id
        ]
        relation_polarity = getattr(parsed_query, "relation_polarity", None)
        spatial_relations: list[str] = []
        if target_results and anchor_results and relation_polarity:
            spatial_relations = build_spatial_relations(
                target_results,
                anchor_results,
                objects_by_id,
                scene_center,
            )
        context = _context(
            query,
            parsed_query,
            target_results,
            anchor_results,
            frame_chunks,
            spatial_relations,
        ) if parsed_query else _context(
            query,
            type("FallbackParse", (), {"target_object": target, "anchor_objects": anchors})(),
            target_results,
            anchor_results,
            frame_chunks,
            spatial_relations,
        )
        prepared.append({
            "row": row,
            "query": query,
            "parse_error": None if parse_error is None else f"{type(parse_error).__name__}: {parse_error}",
            "target_results": target_results,
            "context": context,
            "images": _load_frame_images(frame_chunks, args.max_frame_images) if args.include_frame_images else None,
        })

    selected = _resumable_structured_batch(
        gpt,
        run_structured_batch,
        [item["context"] for item in prepared],
        images_list=[item["images"] for item in prepared] if args.include_frame_images else None,
        response_model=ObjectSelection,
        model=args.selection_model,
        instructions=selection_instructions,
        reasoning_effort="medium",
        detail="high",
        cache_path=cache_prefix.with_name(cache_prefix.name + "_selection_cache.json"),
        batch_size=request_batch_size,
    )

    records = []
    errors = 0
    for item, selection in zip(prepared, selected):
        selection_error = None
        selected_id = None
        if isinstance(selection, Exception):
            errors += 1
            selection_error = f"{type(selection).__name__}: {selection}"
        else:
            selected_id = getattr(selection, "object_id", None) or getattr(selection, "guess_id", None)
        candidates = ordered_candidates(selected_id, item["target_results"], objects_by_id, args.top_k)
        evidence = None
        if candidates:
            selected_node = objects_by_id.get(str(candidates[0]["id"]))
            if selected_node is not None and args.evidence_dir:
                evidence = _write_node_evidence(
                    selected_node,
                    str(item["row"]["query_id"]),
                    Path(args.evidence_dir),
                )
        records.append({
            "schema_version": 1,
            "method": "keysg_released_rag",
            "key": item["row"]["key"],
            "query_id": item["row"]["query_id"],
            "scene_id": item["row"]["scene_id"],
            "query": item["query"],
            "prediction": candidates[0] if candidates else None,
            "candidates": candidates,
            "predicted_node_id": candidates[0]["id"] if candidates else None,
            "top1_evidence": evidence,
            "label_source": "keysg_map_node_label",
            "adapter_error": item["parse_error"] or selection_error,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    # Keep the line-oriented debug artifact and emit the same wrapper consumed
    # by the real-robot VLM adjudication tools used by other methods.
    payload = {
        "metadata": {
            "method": "keysg_released_rag",
            "run_id": args.scene_id,
            "query_count": len(records),
            "predictions_found": sum(bool(r.get("prediction")) for r in records),
            "evidence_ready": sum(bool(r.get("top1_evidence")) for r in records),
            "parser_model": args.parser_model,
            "selection_model": args.selection_model,
            "embedding_model": args.embedding_model,
        },
        "queries": records,
    }
    wrapper_path = output.with_name("retrieval_results.json")
    wrapper_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"scene_id": args.scene_id, "queries": len(records), "selection_errors": errors, "output": str(output), "wrapper": str(wrapper_path)}))
    if errors > args.max_errors:
        raise RuntimeError(f"KeySG selection errors {errors} exceed allowed {args.max_errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit HM3DSem-SQR predictions from a released KeySG map")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-dir", required=True, help="One KeySG output directory (ending in HMP3D/ep_000)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--raw-candidate-k", type=int, default=10)
    parser.add_argument("--frame-candidate-k", type=int, default=10)
    parser.add_argument("--max-frame-images", type=int, default=4)
    parser.add_argument("--include-frame-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parser-model", default="gpt-5-nano")
    parser.add_argument("--selection-model", default="gpt-5-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--evidence-dir", default=None,
                        help="Directory for method-blind boxed Top-1 evidence images")
    parser.add_argument("--max-errors", type=int, default=0)
    parser.add_argument("--request-batch-size", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
