#!/usr/bin/env python3
"""Inference script for Prunnerai test v.1.0"""

import argparse
import json
import os

MODEL_NAME = "Prunnerai test v.1.0"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", MODEL_NAME, "config.json")

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def run_ollama(prompt, config):
    import requests
    r = requests.post("http://localhost:11434/api/generate", json={"model": MODEL_NAME, "prompt": prompt, "system": config.get("system_prompt", ""), "stream": False})
    print(r.json().get("response", ""))

def run_vllm(prompt, config):
    from vllm import LLM, SamplingParams
    llm = LLM(model=config.get("source_url", MODEL_NAME))
    out = llm.generate([prompt], SamplingParams(temperature=0.7, max_tokens=4096))
    for o in out: print(o.outputs[0].text)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "vllm"], default="ollama")
    p.add_argument("--prompt", default="Hello!")
    a = p.parse_args()
    cfg = load_config()
    {"ollama": run_ollama, "vllm": run_vllm}[a.backend](a.prompt, cfg)
