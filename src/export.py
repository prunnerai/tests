#!/usr/bin/env python3
"""
Export model weights for Prunnerai test v.1.0
Supports: SafeTensors, GGUF (via llama.cpp)
Automatically updates weights/manifest.json after export.
"""

import argparse
import hashlib
import json
import os
import torch


MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "manifest.json")


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_manifest(exported_files):
    """Update weights/manifest.json with exported file metadata."""
    from datetime import datetime, timezone

    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

    entries = []
    total = 0
    for path in exported_files:
        size = os.path.getsize(path)
        total += size
        entries.append({
            "path": os.path.basename(path),
            "size_bytes": size,
            "sha256": _file_sha256(path),
        })

    manifest["files"] = entries
    manifest["total_size_bytes"] = total
    manifest["exported_at"] = datetime.now(timezone.utc).isoformat()

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 Updated weights/manifest.json ({len(entries)} files, {total / 1e6:.1f} MB)")


def export_safetensors(checkpoint_path, output_dir="weights/"):
    from safetensors.torch import save_file
    from src.model import build_model

    os.makedirs(output_dir, exist_ok=True)
    model = build_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    state = model.state_dict()
    output_path = os.path.join(output_dir, "Prunnerai test v.1.0.safetensors")
    save_file(state, output_path)
    print(f"✅ Exported to {output_path}")
    _update_manifest([output_path])


def export_gguf(checkpoint_path, output_dir="weights/", quantization="q4_k_m"):
    print("GGUF export requires llama.cpp convert tools.")
    print("1. Export to SafeTensors first")
    print("2. Use llama.cpp/convert.py to convert to GGUF")
    print(f"   python convert.py {output_dir}/Prunnerai test v.1.0.safetensors --outtype {quantization}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--format", choices=["safetensors", "gguf"], default="safetensors")
    parser.add_argument("--quantization", default="q4_k_m")
    args = parser.parse_args()

    if args.format == "safetensors":
        export_safetensors(args.checkpoint)
    else:
        export_gguf(args.checkpoint, quantization=args.quantization)
