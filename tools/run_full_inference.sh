#!/usr/bin/env bash
# Full MMAE inference runner for the agent backup solution.
# Must be run INSIDE the sandbox after setup_autodl.sh.
set -euo pipefail

SANDBOX="/root/autodl-tmp/minimax"
PROJECT="${SANDBOX}/aec-2027-transfer-agent"
VENV="${SANDBOX}/.venv"
HF_CACHE="${SANDBOX}/hf_cache"
MMAE_DIR="${SANDBOX}/mmae"
OUT="${PROJECT}/outputs/full"
mkdir -p "${OUT}"

cd "${PROJECT}"
source "${VENV}/bin/activate"

# GPU detection
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "[gpu] ${GPU_NAME}"
    # Reinstall CUDA PyTorch if needed
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet \
        "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
        --index-url https://download.pytorch.org/whl/cu121
else
    echo "[gpu] CPU-only -- skipping CUDA install"
fi

export HF_HOME="${HF_CACHE}"
export HF_ENDPOINT=https://hf-mirror.com

# Step 1: ensure MMAE data downloaded
if [ ! -f "${MMAE_DIR}/MMAE-meta.json" ]; then
    echo "[mmae] downloading data ..."
    python - <<'PY'
import os
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
out = os.environ.get("MMAE_DIR", "/root/autodl-tmp/minimax/mmae")
os.makedirs(out, exist_ok=True)
snapshot_download(
    repo_id="BoJack/MMAE",
    repo_type="dataset",
    local_dir=out,
    allow_patterns=["*.json", "*.wav", "*.mp3", "*.flac", "*.txt"],
    max_workers=8,
)
print("[mmae] OK")
PY
fi

# Step 2: build fixture (or use full set)
python tools/make_fixture.py --out "${MMAE_DIR}" --n 5

# Step 3: run agent on full MMAE
python -m agent.router \
    --meta "${MMAE_DIR}/MMAE-meta.json" \
    --output_dir "${OUT}"

echo "[done] outputs at ${OUT}"
ls -la "${OUT}"
ls -la "${OUT}/audio" 2>/dev/null || true
ls -la "${OUT}/traces" 2>/dev/null || true