# GPU Training Container
# Framework: unsloth
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default: run training
CMD ["python3", "src/train.py"]
