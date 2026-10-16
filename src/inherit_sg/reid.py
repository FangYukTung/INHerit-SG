"""Second-stage association on spatially and semantically compatible candidates."""
import numpy as np


def choose_visual_match(feature, candidate_indices, features, threshold):
    """Return a candidate index only if its cosine similarity exceeds threshold."""
    if not candidate_indices:
        return None
    vector = np.asarray(feature, dtype=float).reshape(-1)
    values = np.asarray(features, dtype=float)[candidate_indices]
    denom = np.linalg.norm(values, axis=1) * np.linalg.norm(vector)
    scores = np.divide(values @ vector, denom, out=np.full(len(values), -np.inf), where=denom > 0)
    best = int(np.argmax(scores))
    return candidate_indices[best] if scores[best] > threshold else None
