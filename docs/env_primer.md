# Environment Primer: Setup and Running Scripts

## What you need installed on your machine

- **Docker Desktop** (Mac/Windows) or **Docker Engine** (Linux)
  - Mac: https://docs.docker.com/desktop/install/mac-install/
  - Linux: `sudo apt install docker.io` then `sudo usermod -aG docker $USER` (log out and back in)
- **make** — already present on Mac/Linux. On Windows, use WSL2 or run the `docker` commands directly.
- **NVIDIA Container Toolkit** — only needed if you want GPU passthrough on Linux:
  ```bash
  sudo apt install nvidia-container-toolkit
  sudo systemctl restart docker
  ```
  Not needed on Mac (no NVIDIA GPU support) or if running CPU-only.

That's it. Python, CUDA, and all modeling dependencies are inside the image — nothing else goes on your machine.

---

## Project directory layout

Unzip the environment files and put your code alongside them:

```
antisaccade_maturation/
├── Dockerfile
├── Makefile
├── requirements.txt
├── convert_papers.py
├── code/               ← your scripts go here (train.py, analysis.py, etc.)
├── papers/             ← PDFs to convert
├── converted/          ← converted markdown output
└── .cache/             ← auto-created; holds marker + HF model weights
```

The container mounts your entire `antisaccade_maturation/` directory at `/workspace` inside the container, so `/workspace/code/train.py` is your `code/train.py`. Outputs written to `/workspace/outputs/` from inside the container appear in `antisaccade_maturation/outputs/` on your machine. Nothing is lost when the container stops.

---

## Step 1: Build the image

From inside the `antisaccade_maturation/` directory:

```bash
make build
```

This runs `docker build -t antisaccade .` and installs everything — Python, PyTorch, JAX, marker, scikit-learn, the full stack. Takes 10–20 minutes the first time depending on your connection. Subsequent builds are fast because Docker caches layers; only layers below a changed line re-run.

Rebuild only when you change `requirements.txt` or `Dockerfile`. Not between runs.

---

## Step 2: Run a script

```bash
make run SCRIPT=code/train.py
```

Expands to:
```bash
docker run --rm \
    -v $(PWD):/workspace \
    -v $(PWD)/.cache:/workspace/.cache \
    antisaccade \
    /workspace/code/train.py
```

**With GPU:**
```bash
make gpu SCRIPT=code/train.py
```

**Passing arguments to your script:**

The `make` targets don't handle arguments directly. For anything beyond a bare script name, call `docker run` directly:

```bash
docker run --rm \
    -v $(PWD):/workspace \
    -v $(PWD)/.cache:/workspace/.cache \
    antisaccade \
    /workspace/code/train.py --maturation 0.0 --epochs 500 --output outputs/young
```

With GPU:
```bash
docker run --rm --gpus all \
    -v $(PWD):/workspace \
    -v $(PWD)/.cache:/workspace/.cache \
    antisaccade \
    /workspace/code/train.py --maturation 1.0 --epochs 500 --output outputs/adult
```

---

## Step 3: Convert papers (one-time)

```bash
docker run --rm \
    -v $(PWD):/workspace \
    -v $(PWD)/.cache:/workspace/.cache \
    antisaccade \
    /workspace/convert_papers.py --input docs/crucial_literature/ --output docs/crucial_literature/
```

The first run downloads marker's model weights (~1–2 GB) into `.cache/`. Every subsequent run is instant because the cache is mounted as a volume and persists.

---

## Interactive shell

If you want to poke around, run commands manually, or debug:

```bash
make shell
```

Drops you into bash inside the container with your full directory mounted. Your files are live — edits you make to `code/train.py` from your editor on the host are immediately visible inside the container. Exit with `Ctrl+D` or `exit`.

---

## Verifying the environment

From inside a shell (`make shell`), or as a one-off:

```bash
docker run --rm \
    -v $(PWD):/workspace \
    -v $(PWD)/.cache:/workspace/.cache \
    antisaccade - << 'EOF'
import torch, jax, numpy, scipy, sklearn, optax
print(f"torch:   {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
print(f"jax:     {jax.__version__}  |  backend: {jax.default_backend()}")
print(f"numpy:   {numpy.__version__}")
print(f"scipy:   {scipy.__version__}")
print(f"optax:   {optax.__version__}")
EOF
```

On CPU you'll see `CUDA: False` and `backend: cpu`. On a machine with a GPU and `--gpus all`, you'll see `CUDA: True` and `backend: gpu`. Your training scripts don't need to change between the two.

---

## When to rebuild vs when to just re-run

| Situation | Action |
|---|---|
| Changed `requirements.txt` | `make build` |
| Changed `Dockerfile` | `make build` |
| Changed your Python script | Just `make run` — no rebuild |
| Added a new paper to `papers/` | Just re-run `convert_papers.py` |
| Pulled new code from git | Just `make run` — no rebuild unless dependencies changed |

---

## Common issues

**`permission denied` on Linux when running docker:**
```bash
sudo usermod -aG docker $USER
# log out and back in
```

**`docker: Error response from daemon: could not select device driver "nvidia"`:**
NVIDIA Container Toolkit is not installed or Docker wasn't restarted after install. See Step 0 above.

**Script can't find a file:**
Check the path. Inside the container your project root is `/workspace`. So `open("data/trials.npy")` in your script looks for `/workspace/data/trials.npy`, which is `antisaccade_maturation/data/trials.npy` on your machine.

**`.cache/` fills up disk space:**
The marker and HuggingFace weights live in `.cache/` and are a few GB. This is intentional — it means you never re-download them. If you want to clear them: `rm -rf .cache/`.

**Build fails on a specific pip install:**
Run `make shell` and try the failing `pip install` command manually to see the full error. Common causes: version conflicts in `requirements.txt`, or a package that needs a system library not in the image (add it to the `apt-get install` block in the Dockerfile).
