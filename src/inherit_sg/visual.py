"""Fail-open, listwise reranking from stored best-view crops."""
import base64
import io
from pathlib import Path

from PIL import Image, ImageDraw

from .query import decode_model_json, visual_attributes


def evidence_image(obj, context_scale=2.0):
    path, bbox = obj.get("full_image_path"), obj.get("bbox")
    if not path or not Path(path).is_file() or not bbox or len(bbox) != 4:
        return None
    with Image.open(path) as source:
        image = source.convert("RGB")
    x1, y1, x2, y2 = map(float, bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    # Existing reranker convention: pad each side by context_scale * box size.
    left = max(0, int(x1 - context_scale * (x2 - x1)))
    top = max(0, int(y1 - context_scale * (y2 - y1)))
    right = min(image.width, int(x2 + context_scale * (x2 - x1)))
    bottom = min(image.height, int(y2 + context_scale * (y2 - y1)))
    if right <= left or bottom <= top:
        return None
    crop = image.crop((left, top, right, bottom))
    ImageDraw.Draw(crop).rectangle((x1-left, y1-top, x2-left, y2-top), outline="red", width=4)
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class VisualReranker:
    def __init__(self, client, model, confidence=0.8, budget=5):
        self.client, self.model, self.confidence, self.budget = client, model, confidence, budget

    def rerank(self, instruction, query, pool, memory):
        trace = {"eligible": False, "promoted": False}
        if not pool or not visual_attributes(query):
            return pool, trace
        visible = []
        for node_id in pool[:self.budget]:
            try:
                image = evidence_image(memory.objects[node_id])
            except (OSError, ValueError):
                image = None
            if image:
                visible.append((node_id, image))
        if len(visible) < 2 or visible[0][0] != pool[0]:
            return pool, trace
        trace["eligible"] = True
        content = [{"type": "text", "text": (
            "Select the red-box object matching the instruction and its visible attributes. "
            "Ignore unrelated background. Return JSON {choice: candidate_id or UNCERTAIN, "
            "confidence: number from 0 to 1}. Do not guess when evidence is inconclusive.\n"
            f"Instruction: {instruction}\nPositive visual attributes: {visual_attributes(query)}"
        )}]
        for node_id, image in visible:
            content.extend([{"type": "text", "text": f"Candidate ID: {node_id}"},
                            {"type": "image_url", "image_url": {"url": image}}])
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=1536,
                messages=[{"role": "user", "content": content}], response_format={"type": "json_object"},
            )
            payload = decode_model_json(response.choices[0].message.content)
            choice = payload["choice"]
            confidence = float(payload["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError("invalid confidence")
            trace.update(choice=choice, confidence=confidence)
            if choice in {node_id for node_id, _ in visible} and confidence >= self.confidence and choice != pool[0]:
                trace["promoted"] = True
                return [choice] + [node_id for node_id in pool if node_id != choice], trace
        except Exception as error:
            # Preserve the pool; do not expose endpoint credentials in error text.
            trace["error_type"] = type(error).__name__
        return pool, trace
