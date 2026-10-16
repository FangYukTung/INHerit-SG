"""Public mapping, query, and textual-memory export entrypoints."""
import argparse
import json
import os
from pathlib import Path

import yaml


def load_config(path):
    value = yaml.safe_load(Path(path).read_text())
    def expand(item):
        if isinstance(item, str):
            result = os.path.expandvars(item)
            if "${" in result:
                raise ValueError(f"Unset environment variable in configuration value: {item}")
            return result
        if isinstance(item, list):
            return [expand(v) for v in item]
        if isinstance(item, dict):
            if "api_key" in item:
                raise ValueError("Use api_key_env rather than storing credentials in configuration")
            return {k: expand(v) for k, v in item.items()}
        return item
    return expand(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mapping = sub.add_parser("map")
    mapping.add_argument("--config", required=True)
    query = sub.add_parser("query")
    query.add_argument("--config", required=True)
    query.add_argument("--map", required=True)
    query.add_argument("--text", required=True)
    query.add_argument("--constraints", help="Validated query JSON; skips model parsing")
    query.add_argument("--no-visual", action="store_true")
    export = sub.add_parser("export-memory")
    export.add_argument("--map", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "map":
        from .mapping import run_mapping
        run_mapping(load_config(args.config))
        return
    from .retrieval import Memory, RetrievalSettings, StructuredRetriever
    memory = Memory.load(args.map)
    if args.command == "export-memory":
        with Path(args.output).open("x") as handle:
            handle.write(memory.text() + "\n")
        return
    from .query import parse_object
    from .inference import query_components
    config = load_config(args.config)
    parser, reranker = query_components(config, parse=not args.constraints, visual=not args.no_visual)
    parsed = parse_object(json.loads(Path(args.constraints).read_text())) if args.constraints else parser.parse(args.text)
    retriever = StructuredRetriever(memory, settings=RetrievalSettings(**config.get("retrieval", {})))
    print(json.dumps(retriever.retrieve(args.text, parsed, reranker), indent=2))


if __name__ == "__main__":
    main()
