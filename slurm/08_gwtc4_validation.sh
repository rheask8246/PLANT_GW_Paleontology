#!/bin/bash
#SBATCH --job-name=gwtc4_val
#SBATCH --output=logs/08_gwtc4_validation.%j.out
#SBATCH --error=logs/08_gwtc4_validation.%j.err
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --constraint=lustre
#SBATCH --time=12:00:00
#SBATCH --account=sdp153
#SBATCH --export=ALL
#SBATCH --no-requeue

# ACCESS Expanse (SDSC): https://www.sdsc.edu/systems/expanse/user_guide.html
#
# - gpu-shared: fractional GPU node (vs gpu = full node, 4× GPU SU + 128 CPU SU style billing).
# - constraint=lustre: REQUIRED if reading/writing under /expanse/lustre/*; jobs without it may land
#   on nodes without Lustre and fail (see "Submitting Jobs Using Lustre" in the user guide).
# - Default 1 GPU gives 1 CPU + 1G unless you request more; we request 8 CPUs for NumPy/CPU work
#   and mem for checkpoints + tables (raise --mem if you OOM).
# - gpu-shared billing uses max(cores, memory fraction) vs node capacity; keep requests tight.
# - no-requeue: avoid silent restarts overwriting outputs on node failures (user guide).
#
# Walltime: light defaults ~1–3 GPU hours; --paper-quality needs much more MC (often 8–24+ h).
# Raise --time or use GWTC4_PAPER_QUALITY=0 with manual GWTC4_NROWS / GWTC4_NEVENTS if you hit limit.
# Max for gpu-shared is 48h.
#
# Usage (from repo root PLANT_GW_Paleontology/, where hyperparam_table_encoded.csv lives):
#   sbatch slurm/08_gwtc4_validation.sh
#
# Paper-like smooth figure (recommended once checkpoints exist; needs scipy in env):
#   sbatch --export=ALL,GWTC4_PAPER_QUALITY=1 slurm/08_gwtc4_validation.sh
#
# Optional overrides (examples):
#   sbatch --export=ALL,GWTC4_USE_TEX=1 slurm/08_gwtc4_validation.sh   # requires full TeX (type1cm, etc.)
#   sbatch --export=ALL,GWTC4_NBOOT=4,GWTC4_NROWS=128,GWTC4_NEVENTS=128 slurm/08_gwtc4_validation.sh
#   sbatch --export=ALL,GWTC4_OUT=plots/gwtc4_validation/run_job.pdf slurm/08_gwtc4_validation.sh
# For longer paper-quality runs, edit #SBATCH --time above (gpu-shared allows up to 48h).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Use cluster slurm.conf when present (avoids configless DNS SRV failures on some nodes).
[[ -r /etc/slurm/slurm.conf ]] && export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"

# Lmod in non-login batch jobs (Expanse user guide).
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh

# Do not mix `module load cpu` with `module load gpu` on Expanse.
module purge
module load gpu

# Prefer a local CUDA-enabled venv if present; otherwise conda env `plant` (as in slurm/04_*.sh).
if [[ -f ".venv311/bin/activate" ]]; then
  source ".venv311/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
else
  CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate plant
fi

# TinyTeX (matplotlib usetex): latex + dvipng must be on PATH (batch nodes may not match login PATH).
export PATH="${HOME}/.TinyTeX/bin/x86_64-linux:${HOME}/bin:${PATH:-}"

# Align host-side BLAS / NumExpr with Slurm CPUs (single Python process).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== Host: $(hostname) ==="
echo "=== CWD:  $PWD ==="
echo "=== Python: $(python --version) ==="
echo "=== SLURM: partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?} gpus=${SLURM_GPUS_ON_NODE:-?} ==="
echo "=== Start: $(date -Is) ==="

DEVICE="${GWTC4_DEVICE:-cuda}"
NBOOT="${GWTC4_NBOOT:-12}"
NROWS="${GWTC4_NROWS:-256}"
NEVENTS="${GWTC4_NEVENTS:-256}"
NBINS="${GWTC4_NBINS:-60}"
MMAX="${GWTC4_MMAX:-180.0}"
SEED="${GWTC4_SEED:-42}"
OUT_PATH="${GWTC4_OUT:-}"

# Compute nodes often lack a full TeX stack (TinyTeX missing type1cm.sty breaks matplotlib usetex).
# Default: mathtext only. Set GWTC4_USE_TEX=1 if you have a complete LaTeX install on the batch node.
NO_TEX_FLAG="--no-tex"
if [[ "${GWTC4_USE_TEX:-0}" == "1" ]]; then
  NO_TEX_FLAG=""
fi

PAPER_FLAG=()
if [[ "${GWTC4_PAPER_QUALITY:-0}" == "1" ]]; then
  PAPER_FLAG=(--paper-quality)
  echo "=== GWTC4_PAPER_QUALITY=1 (high MC + smoothing + auto limits + m1*m2 scale) ==="
fi

OUT_FLAG=()
if [[ -n "${OUT_PATH}" ]]; then
  OUT_FLAG=(--out "${OUT_PATH}")
fi

EXTRA_PY_ARGS=()
# Space-separated extra args, e.g. GWTC4_EXTRA='--rate-q-hi 0.99 --smooth-sigma 1.5'
if [[ -n "${GWTC4_EXTRA:-}" ]]; then
  read -r -a EXTRA_PY_ARGS <<< "${GWTC4_EXTRA}"
fi

# -u = unbuffered stdout/stderr so logs update during long runs.
python -u gwtc4_validation.py \
  --device "${DEVICE}" \
  ${NO_TEX_FLAG} \
  "${PAPER_FLAG[@]}" \
  "${OUT_FLAG[@]}" \
  --n-boot "${NBOOT}" \
  --n-rows "${NROWS}" \
  --n-events-per-row "${NEVENTS}" \
  --nbins "${NBINS}" \
  --mmax "${MMAX}" \
  --seed "${SEED}" \
  "${EXTRA_PY_ARGS[@]}"

echo "=== End: $(date -Is) ==="
