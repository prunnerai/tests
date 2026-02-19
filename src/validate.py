#!/usr/bin/env python3
"""
Config validation for Prunnerai test v.1.0
Checks config.json against required fields and valid values.
"""

import json
import sys
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Prunnerai test v.1.0", "config.json")

VALID_ARCHITECTURES = ["transformer", "mamba", "rwkv", "moe", "custom"]
VALID_FRAMEWORKS = ["pytorch", "unsloth", "transformers", "axolotl", "jax"]

def validate():
    if not os.path.exists(CONFIG_PATH):
        print(f"Config not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    errors = []

    if not config.get("name"):
        errors.append("Missing required field: name")

    scratch = config.get("scratch", config.get("build_config", {}).get("scratch", {}))
    if scratch:
        arch = scratch.get("architecture", "")
        if arch and arch not in VALID_ARCHITECTURES:
            errors.append(f"Invalid architecture: {arch}. Must be one of {VALID_ARCHITECTURES}")

        fw = scratch.get("framework", "")
        if fw and fw not in VALID_FRAMEWORKS:
            errors.append(f"Invalid framework: {fw}. Must be one of {VALID_FRAMEWORKS}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)
    else:
        print("Config validation passed")

if __name__ == "__main__":
    validate()
