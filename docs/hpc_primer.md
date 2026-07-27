# HPC Primer: Running the Antisaccade Environment on a Cluster

## The core problem Docker solves on HPC

HPC clusters run many users' jobs on shared hardware. You don't have root access, you can't install system packages, and the Python/CUDA versions on the cluster may differ from your local machine. Docker solves this by bundling your entire environment — Python, CUDA, PyTorch, JAX, marker, everything — into a single image file that runs identically everywhere.

The catch: most HPC clusters don't allow Docker directly (it requires root). They use **Singularity** (now called Apptainer), which runs containers without root. Singularity can convert a Docker image directly.

---

## Workflow overview

```
Local machine                      HPC cluster
─────────────────                  ──────────────────────────────
docker build -t antisaccade .  →   singularity build antisaccade.sif ...
                               →   sbatch job.sh   (runs your script inside .sif)
```

You build once locally, transfer the `.sif` file to the cluster, and submit jobs.

---

## Step 1: Build the Docker image locally

```bash
# In the directory containing your Dockerfile
docker build -t antisaccade .
```

This only needs to run again if you change `requirements.txt` or the `Dockerfile`.

---

## Step 2: Convert to a Singularity image

Singularity reads Docker images in two ways depending on what's available to you.

**Option A — from your local Docker daemon** (if Singularity is installed locally):
```bash
singularity build antisaccade.sif docker-daemon://antisaccade:latest
```

**Option B — push to Docker Hub first, pull on the cluster** (most common):
```bash
# Local: push to Docker Hub (free account required)
docker tag antisaccade yourusername/antisaccade:latest
docker push yourusername/antisaccade:latest

# On the cluster: pull and convert
singularity build antisaccade.sif docker://yourusername/antisaccade:latest
```

**Option C — build directly on the cluster** (if your cluster allows it):
```bash
# Transfer Dockerfile + requirements.txt to cluster, then:
singularity build antisaccade.sif docker-daemon://antisaccade:latest
# Note: this requires Docker on the cluster, which is rare.
# More likely you'll use option B.
```

The `.sif` file is a single portable file (~5–8 GB). Transfer it to the cluster:
```bash
scp antisaccade.sif yourname@cluster.university.edu:/scratch/yourname/
```

---

## Step 3: Test interactively on the cluster

Before submitting batch jobs, verify the environment works in an interactive session:

```bash
# Request an interactive node (syntax varies by cluster — this is SLURM)
srun --pty --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 bash

# CPU run
singularity exec \
    --bind /scratch/yourname:/workspace \
    /scratch/yourname/antisaccade.sif \
    python /workspace/code/train.py

# GPU run (--nv passes through NVIDIA GPU)
singularity exec --nv \
    --bind /scratch/yourname:/workspace \
    /scratch/yourname/antisaccade.sif \
    python /workspace/code/train.py
```

`--bind /scratch/yourname:/workspace` mounts your scratch directory into the container at `/workspace`, the same path your scripts expect. Your code, data, and outputs live there on the cluster filesystem and persist after the job ends.

---

## Step 4: Submit a batch job

Save this as `job.sh` in your project directory on the cluster, editing the SLURM directives for your institution's queue names and limits.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=antisaccade
#SBATCH --output=logs/%j.out        # stdout — %j is the job ID
#SBATCH --error=logs/%j.err         # stderr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00             # wall time limit HH:MM:SS
#SBATCH --partition=gpu             # GPU partition name (varies by cluster)
#SBATCH --gres=gpu:1                # request 1 GPU

# Paths — edit these for your cluster
SIF=/scratch/$USER/antisaccade.sif
WORKSPACE=/scratch/$USER

mkdir -p $WORKSPACE/logs

singularity exec --nv \
    --bind $WORKSPACE:/workspace \
    $SIF \
    python /workspace/code/train.py \
        --maturation_scalar 0.0 \
        --output_dir /workspace/outputs/young

# Remove --nv and --partition=gpu / --gres=gpu:1 for a CPU-only job
```

Submit and monitor:
```bash
sbatch job.sh
squeue --me               # check job status
scancel <JOBID>           # cancel if needed
tail -f logs/<JOBID>.out  # stream output live
```

---

## CPU vs GPU: where the decision actually lives

The Docker image is CUDA-capable regardless. Whether a job uses a GPU is determined by two things:

1. **Whether you pass `--nv`** (Singularity) or `--gpus all` (Docker) at runtime
2. **Whether your script checks and uses the device**

In your training scripts, add this near the top — it requires no changes between local and HPC runs:

```python
import torch
import jax

# PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {device}")

# JAX
print(f"JAX backend: {jax.default_backend()}")  # 'gpu', 'tpu', or 'cpu'
```

That's it. The script is identical; the runtime environment determines what hardware it sees.

---

## Caching model weights on HPC

Marker and HuggingFace weights are downloaded on first use. On HPC you want these in your scratch space, not re-downloaded for every job.

The `Dockerfile` sets:
```
ENV HF_HOME=/workspace/.cache/huggingface
ENV MARKER_CACHE_DIR=/workspace/.cache/marker
```

So as long as you `--bind` your scratch directory to `/workspace`, weights downloaded in one job are reused in all subsequent jobs automatically. The first job will be slower; every subsequent one is instant.

---

## Cluster-specific notes

Most clusters have documentation for Singularity/Apptainer. The commands above are standard SLURM, but a few things vary:

| Thing that varies | Where to check |
|---|---|
| Partition names (`gpu`, `high-mem`, etc.) | `sinfo` command or cluster docs |
| GPU request syntax (`--gres=gpu:1` vs `--gres=gpu:a100:1`) | Cluster docs |
| Scratch directory path (`/scratch`, `/work`, `/project`) | Cluster docs |
| Whether Docker Hub pull is allowed | Cluster docs / IT |
| Singularity vs Apptainer command name | Try both; Apptainer is the newer rename |

If your cluster uses PBS/Torque instead of SLURM, the `#SBATCH` directives become `#PBS` directives, but the `singularity exec` command is identical.
