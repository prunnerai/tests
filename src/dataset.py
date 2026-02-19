#!/usr/bin/env python3
"""
Dataset loader for Prunnerai test v.1.0
Supports: JSONL (instruction tuning), Parquet, text files
"""

import json
import os
from pathlib import Path

try:
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


def load_dataset(data_dir="data/", tokenizer=None, max_length=8192):
    """
    Load training data from the data/ directory.
    Supports .jsonl files with the format:
      {"instruction": "...", "input": "...", "output": "..."}
    or
      {"messages": [{"role": "user", "content": "..."}, ...]}
    """
    data_path = Path(data_dir)
    all_items = []

    for f in sorted(data_path.glob("*.jsonl")):
        all_items.extend(load_jsonl(f))

    if not all_items:
        print(f"⚠️  No .jsonl files found in {data_dir}")
        print("   Create data/train.jsonl with instruction-tuning data.")
        # Return a minimal dummy dataset for testing
        all_items = [{"instruction": "Hello", "input": "", "output": "Hi there!"}]

    print(f"Loaded {len(all_items)} examples from {data_dir}")
    return all_items


if __name__ == "__main__":
    data = load_dataset()
    print(f"Dataset size: {len(data)}")
    if data:
        print(f"Sample: {json.dumps(data[0], indent=2)[:500]}")
