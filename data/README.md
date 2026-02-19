# Training Data

Place your training data in this directory.

## Supported Formats

### Instruction Tuning (JSONL)
```jsonl
{"instruction": "Summarize the following text", "input": "Long article...", "output": "Summary..."}
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
```

### Chat Format (JSONL)
```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

### DPO / Preference Data (JSONL)
```jsonl
{"prompt": "Explain gravity", "chosen": "Gravity is...", "rejected": "I don't know..."}
```

## File Naming
- `train.jsonl` — Main training data (required)
- `eval.jsonl` — Evaluation data (optional)

## Notes
- UTF-8 encoding required
- One JSON object per line
- Recommended: 1,000+ examples for fine-tuning, 10,000+ for training from scratch
