# ─────────────────────────────────────────────────────────────────────────────
# Antisaccade maturation modeling environment
#
# One CUDA-capable image. GPU is used automatically if available at runtime;
# falls back to CPU silently. Same image works locally and on HPC.
#
# Build (once, or when dependencies change):
#   docker build -t antisaccade .
#
# Run a script:
#   docker run --rm -v $(pwd):/workspace antisaccade python train.py
#
# With GPU (if available):
#   docker run --rm --gpus all -v $(pwd):/workspace antisaccade python train.py
#
# Convert to Singularity for HPC:
#   singularity build antisaccade.sif docker-daemon://antisaccade:latest
#   singularity exec --nv antisaccade.sif python train.py
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        git \
        wget \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# PyTorch is required for marker/surya; avoid torchvision because transformers
# can import it eagerly and fail on mismatched wheel builds.
RUN pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cu121

# JAX: same principle — jax[cuda12] uses GPU if present, CPU otherwise.
RUN pip install --no-cache-dir "jax[cuda12_pip]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Everything else
RUN pip install --no-cache-dir \
        $(grep -v -E "^(torch|torchvision|jax|jaxlib|#|$)" requirements.txt)

# code-review-graph: installed so Copilot/Claude Code can use it inside this env.
# You don't invoke it directly; your AI coding assistant does.
RUN pip install --no-cache-dir "code-review-graph[communities,embeddings]"

# ── Model weight cache ────────────────────────────────────────────────────────
# marker and HuggingFace weights land here. Mount as a volume to persist
# across runs so you only download once.
ENV HF_HOME=/workspace/.cache/huggingface
ENV MARKER_CACHE_DIR=/workspace/.cache/marker

ENTRYPOINT ["python"]
