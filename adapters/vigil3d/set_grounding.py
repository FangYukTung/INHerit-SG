"""Set/empty-set selection from retrieved candidates, without target annotations."""
import json

from inherit_sg.visual import evidence_image


def select_set(client, model, instruction, candidates, memory):
    ids = [item.get("node_id", item.get("id")) for item in candidates]
    content = [{"type": "text", "text": (
        "Select all candidate objects satisfying the query, or an empty set when none do. "
        "Return JSON {object_ids: [candidate IDs]}. Do not infer a target count.\n"
        f"Query: {instruction}\nTextual scene memory: {memory.text()}"
    )}]
    for node_id in ids:
        content.append({"type": "text", "text": f"Candidate ID: {node_id}"})
        try:
            image = evidence_image(memory.objects[node_id])
        except (OSError, ValueError):
            image = None
        if image:
            content.append({"type": "image_url", "image_url": {"url": image}})
    try:
        response = client.chat.completions.create(model=model, temperature=0, max_tokens=1536,
            messages=[{"role": "user", "content": content}], response_format={"type": "json_object"})
        selected = json.loads(response.choices[0].message.content)["object_ids"]
        if not isinstance(selected, list) or any(not isinstance(i, str) for i in selected) or len(selected) != len(set(selected)) or not set(selected) <= set(ids):
            raise ValueError("invalid candidate set")
        return selected, {"fallback": False}
    except Exception as error:
        return ids[:1], {"fallback": True, "error_type": type(error).__name__}
