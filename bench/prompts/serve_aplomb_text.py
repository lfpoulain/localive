"""Serve the Aplomb 2 short text checkpoint for the Localive tool-routing benchmark.

    python bench/prompts/serve_aplomb_text.py --model models/decision/aplomb-2

Requires the publisher's local Aplomb 2 Python files, short weights and grouped
manifest. Localive sends at most eight primitive questions per call. The published
grouped checkpoint starts at seventeen, so every test uses the short text weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import HTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18857)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    root = args.model.resolve(strict=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    sys.path.insert(0, str(root))

    import torch
    from aplomb.export2 import load_export
    from aplomb.group2 import served_sizes
    from aplomb.runtime import Model, PackageError, media_nodes
    from serve_aplomb import handler

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = json.loads((root / "grouped" / "a2_export.json").read_text("utf-8"))
    minimum_group = min(served_sizes(manifest["grouped"]))

    class ShortTextModel(Model):
        def adapter(self, which):
            if which != "short":
                raise PackageError("This benchmark serves the short text checkpoint only.")
            if which not in self.adapters:
                self.adapters[which] = load_export(
                    str(self.root), device=self.device, dtype=getattr(torch, self.dtype),
                    audio_tower=None,
                )
            return self.adapters[which]

        def short_rows(self, ad, state, prims, plans, want):
            if len(prims) >= minimum_group or any(media_nodes(state)):
                raise PackageError("This benchmark serves small text-only question sets.")
            return super().short_rows(ad, state, prims, plans, want)

    model = ShortTextModel(root, device=args.device, dtype=args.dtype)
    # The reference would load the grouped weights here merely to reject a smaller batch.
    model.reading = {**model.reading, "grouped": False}
    with HTTPServer(("127.0.0.1", args.port), handler(model)) as server:
        print(f"Aplomb 2 ready on localhost:{args.port} ({args.device} {args.dtype}, "
              f"{args.threads} CPU threads)", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
