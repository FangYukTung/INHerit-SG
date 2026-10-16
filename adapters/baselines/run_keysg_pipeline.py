#!/usr/bin/env python3
"""Start unmodified KeySG while exposing its bundled SAM2 YAML to Hydra."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--sam2-config-root", type=Path, required=True)
    args, hydra_args = parser.parse_known_args()
    if hydra_args[:1] == ["--"]:
        hydra_args = hydra_args[1:]

    upstream = args.upstream.resolve()
    config_root = args.sam2_config_root.resolve()
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # KeySG references `sam2_configs/...` from its own frozen config directory.
    # SAM2's build_sam2 composes configs through package `sam2`, so append the
    # KeySG config parent to that package path without editing either upstream.
    import sam2

    if str(config_root) not in sam2.__path__:
        sam2.__path__.append(str(config_root))

    from keysg_runtime_patches import (
        install_chat_completions_adapter,
        install_embedding_batch_limit,
        install_small_room_keyframe_fallback,
    )

    install_small_room_keyframe_fallback()
    install_embedding_batch_limit()
    if os.environ.get("KEYSG_USE_CHAT_COMPLETIONS_ADAPTER") == "1":
        install_chat_completions_adapter()

    entrypoint = upstream / "main_pipeline.py"
    sys.argv = [str(entrypoint), *hydra_args]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
