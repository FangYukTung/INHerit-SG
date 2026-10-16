import json
from pathlib import Path

def load_split(path):
    return json.loads(Path(path).read_text())

def load_source_query(dataset_root, row):
    rows = json.loads((Path(dataset_root) / row["source_file"]).read_text())
    return next(q for q in rows if q["meta"]["query_id"] == row["query_id"])
