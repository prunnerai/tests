#!/usr/bin/env python3
"""
Training hyperparameter config loader for Prunnerai test v.1.0
Reads from models/Prunnerai test v.1.0/config.json
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Prunnerai test v.1.0", "config.json")


class TrainConfig:
    def __init__(self):
        with open(CONFIG_PATH) as f:
            raw = json.load(f)
        self.raw = raw
        scratch = raw.get("scratch", raw.get("build_config", {}).get("scratch", {}))
        self.architecture = scratch.get("architecture", "transformer")
        self.framework = scratch.get("framework", "pytorch")
        self.model_size = scratch.get("model_size", "7B")
        self.training_algorithms = scratch.get("training_algorithms", ["sft"])
        self.context_window = raw.get("build_config", {}).get("context_window", 8192)
        self.temperature = raw.get("build_config", {}).get("temperature", 0.7)

    def __repr__(self):
        return (f"TrainConfig(arch={self.architecture}, fw={self.framework}, "
                f"size={self.model_size}, algos={self.training_algorithms})")


if __name__ == "__main__":
    cfg = TrainConfig()
    print(cfg)
