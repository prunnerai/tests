#!/usr/bin/env python3
"""
Evaluation harness for Prunnerai test v.1.0
"""

import torch
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Prunnerai test v.1.0", "config.json")


def evaluate(checkpoint_path="checkpoints/final.pt"):
    print(f"Evaluating checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Train first with: make train")
        return

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # Load model
    from src.model import build_model
    model = build_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Model size: {param_count * 2 / 1e9:.2f} GB (fp16)")

    # TODO: Add perplexity evaluation
    # TODO: Add generation quality tests
    print("\n✅ Evaluation complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    args = parser.parse_args()
    evaluate(args.checkpoint)
