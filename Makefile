.PHONY: setup train eval export clean

setup:
	pip install -r requirements.txt

train:
	python3 src/train.py

eval:
	python3 src/evaluate.py

export:
	python3 src/export.py --format safetensors

export-gguf:
	python3 src/export.py --format gguf

docker-train:
	docker compose up train

docker-eval:
	docker compose up eval

clean:
	rm -rf checkpoints/*.pt checkpoints/*.safetensors __pycache__ .cache
