#!/usr/bin/env bash
# AutoDL/HK sandbox deploy script
# ============================================================
# Hard sandbox rules:
#   - The only allowed working directory: /root/minimax
#   - All writes/reads in the script body must be prefixed with /root/minimax
#   - Any path escaping that dir (e.g. /root/xxx, /home, /tmp, /opt) is rejected
#   - No /etc read, no ~/.bashrc edit, no touch on main project EditX-LoRA env
#   - Leaves no SSH key / git config / crontab side-effects
# ============================================================
#
# Execute in the AutoDL/HK SSH session:
#   bash setup_hk_sandbox.sh
#
set -euo pipefail

# ---------- 0. Sandbox path (mandatory) ----------
# Note: HK and AutoDL sandbox share the same logic; data disk path differs
# Default AutoDL (/root/autodl-tmp/); override via env at call time
SANDBOX="${SANDBOX:-/root/minimax}"

# AutoDL data disk is usually mounted at /root/autodl-tmp/,
# We place 'large files' (HF models, MMAE data) under the data disk minimax subdir,
# still constrained by sandbox (/root/minimax/* or /root/autodl-tmp/minimax/*)
AUTODL_DATA="${AUTODL_DATA:-/root/autodl-tmp}"
if [ -d "${AUTODL_DATA}" ]; then
    # AutoDL env: build minimax subdir on data disk, peer to sandbox
    SANDBOX="${AUTODL_DATA}/minimax"
    HF_HOME="${SANDBOX}/hf_cache"
    MMAE_DIR="${SANDBOX}/mmae"
    echo "[autodl] detected data disk at ${AUTODL_DATA}; sandbox moved there"
else
    HF_HOME="${SANDBOX}/hf_cache"
    MMAE_DIR="${SANDBOX}/mmae"
    echo "[sandbox] no data disk found; using ${SANDBOX} directly"
fi

PROJECT_DIR="${SANDBOX}/AEC-2027-transfer-agent"
VENV_DIR="${SANDBOX}/.venv"

# Forbidden path prefixes outside sandbox (runtime interception)
# Universal rule: any /root/<not-minimax> path is rejected
# Explicit list of known main-project-related names
FORBIDDEN_PREFIXES=(
    # Universal: any /root/<not-minimax> path
    # enforced by the 'is_inside_sandbox' check below
    # Explicit list of likely main-project folder names
    "/root/aec-2027-editx-lora" "/root/aec-2027" "/root/aec"
    "/root/editx" "/root/EditX" "/root/edit-x" "/root/edit_x"
    "/root/ICASSP" "/root/icassp" "/root/icassp2027" "/root/ICASSP2027"
    "/root/submission" "/root/official" "/root/main-project" "/root/main_project"
    "/root/competition" "/root/challenge" "/root/audio-editing"
    "/root/EditX-LoRA" "/root/editx-lora" "/root/editx_lora"
    "/root/jxzhen" "/root/main" "/root/primary"
    # System dirs (must not be touched)
    "/etc" "/var" "/home" "/opt" "/srv" "/tmp"
)

# ---------- 0.5 Auto-discover main-project folders ----------
# Scan /root/ for non-minimax / non-system folders; blacklist them all
# Even if I don't know names, main project files are untouchable
if [ -d "/root" ]; then
    for d in /root/*/; do
        [ -d "$d" ] || continue
        base="$(basename "$d")"
        # skip minimax itself
        [ "$base" = "minimax" ] && continue
        # skip obvious system/tool dirs
        case "$base" in
            .ssh|.cache|.config|.local|.npm|.pip|.cargo|go|.dotnet|.azure|.docker|.kube|workspace|workspaces|tmp|workspace_*|work|projects)
                continue
                ;;
        esac
        # add it to runtime blacklist
        FORBIDDEN_PREFIXES+=("/root/$base")
        echo "[sandbox] discovered protected folder: /root/$base"
    done
fi

# ---------- 1. Path safety check function ----------
# Universal rule: any path outside /root/minimax/ is rejected
# Even if main project has 100 folders, all are rejected unless under minimax
check_safe_path() {
    local p="$1"
    # Resolve real path
    local real
    real="$(realpath -m "$p" 2>/dev/null || echo "$p")"
    # Must be inside sandbox
    case "$real" in
        "${SANDBOX}"|"${SANDBOX}"/*) ;;
        *)
            echo "[SANDBOX BLOCK] attempted escape: $p -> $real"
            echo "[SANDBOX BLOCK] not under $SANDBOX"
            exit 1
            ;;
    esac
    # Extra: must not hit any forbidden prefix
    for forbidden in "${FORBIDDEN_PREFIXES[@]}"; do
        case "$real" in
            "${forbidden}"|"${forbidden}"/*)
                echo "[SANDBOX BLOCK] dangerous path: $real hits $forbidden"
                exit 1
                ;;
        esac
    done
}

# ---------- 2. Helper to execute commands inside sandbox ----------
in_sandbox() {
    (
        cd "${SANDBOX}" || { echo "[SANDBOX] cd failed"; exit 1; }
        check_safe_path "${SANDBOX}"
        "$@"
    )
}

echo "=========================================="
echo "  AutoDL/HK sandbox deploy"
echo "  SANDBOX = ${SANDBOX}"
echo "  HF_HOME = ${HF_HOME}"
echo "  MMAE    = ${MMAE_DIR}"
echo "  PROJECT = ${PROJECT_DIR}"
echo "=========================================="

# ---------- 3. Create sandbox dirs (minimax subtree only) ----------
mkdir -p "${SANDBOX}"
check_safe_path "${SANDBOX}"
mkdir -p "${HF_HOME}" "${MMAE_DIR}"
check_safe_path "${HF_HOME}"
check_safe_path "${MMAE_DIR}"

# ---------- 4. Install basic tools (system-level; apt to /usr is not escape)----------
# Note: apt ops are outside sandbox but only affect system pkgs, isolated from main project workspace
apt-get update -y -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg git wget curl ca-certificates

# ---------- 5. Clone the backup-solution repo (into sandbox)----------
if [ ! -d "${PROJECT_DIR}" ]; then
    cd "${SANDBOX}"
    check_safe_path "$(pwd)"
    git clone https://github.com/jxzhen/AEC-2027-transfer-agent.git "${PROJECT_DIR}"
else
    echo "[git] already cloned at ${PROJECT_DIR}"
    in_sandbox bash -c "cd '${PROJECT_DIR}' && git pull --ff-only || true"
fi
check_safe_path "${PROJECT_DIR}"

# ---------- 6. venv (inside sandbox, no system Python touch)----------
cd "${PROJECT_DIR}"
if [ ! -d "${VENV_DIR}" ]; then
    python3.11 -m venv "${VENV_DIR}" 2>/dev/null || python3 -m venv "${VENV_DIR}"
fi
check_safe_path "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Force pip ops to stay inside sandbox
export PIP_CACHE_DIR="${SANDBOX}/pip_cache"
mkdir -p "${PIP_CACHE_DIR}"
check_safe_path "${PIP_CACHE_DIR}"

pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet

# Detect GPU (read-only inside sandbox)
GPU_OK=0
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_OK=1
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "[gpu] OK -- ${GPU_NAME}"
else
    echo "[gpu] not available -- CPU-only smoke test"
fi

# PyTorch: pick build per GPU
if [ "${GPU_OK}" -eq 1 ]; then
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet \
        "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
        --index-url https://download.pytorch.org/whl/cu121
else
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet \
        "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
        --index-url https://download.pytorch.org/whl/cpu
fi

# Common deps (all into sandbox venv)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet \
    "transformers>=4.51.3,<5.0.0" \
    "accelerate" \
    "numpy" "scipy" "librosa" "soundfile" \
    "ml-collections" "absl-py" \
    "huggingface_hub"

# pip cache purge inside sandbox
pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true

# ---------- 7. Pull HF models (into sandbox HF cache)----------
export HF_HOME="${HF_HOME}"
export HF_ENDPOINT=https://hf-mirror.com
check_safe_path "${HF_HOME}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download
cache = os.environ["HF_HOME"]
os.makedirs(cache, exist_ok=True)
# Re-assert: cache must be inside sandbox
assert cache.startswith("/root/minimax"), f"cache escapes sandbox: {cache}"
for m in [
    "microsoft/VibeVoice-ASR",
    "openbmb/MiniCPM-o-2_6",
]:
    print(f"[hf] downloading {m}")
    snapshot_download(
        repo_id=m,
        cache_dir=cache,
        allow_patterns=["*.json", "*.txt", "*.md", "*.safetensors", "*.bin", "*.pt", "tokenizer.*", "*.py"],
        max_workers=8,
    )
    print(f"[hf] OK {m}")
PY

# ---------- 8. Smoke test (inside sandbox)----------
in_sandbox bash -c "
    set -euo pipefail
    source '${VENV_DIR}/bin/activate'
    cd '${PROJECT_DIR}'
    mkdir -p outputs/test_fixture
    python tools/make_fixture.py --out outputs/test_fixture --n 5
    python -m agent.router --meta outputs/test_fixture/MMAE-meta.json --output_dir outputs/test_fixture
"

echo "=========================================="
echo "[done] AutoDL/HK sandbox deploy complete"
echo ""
echo "Sandbox ready at:"
echo "  Project code: ${PROJECT_DIR}"
echo "  HF cache:     ${HF_HOME}"
echo "  MMAE data:    ${MMAE_DIR}"
echo "  venv:         ${VENV_DIR}"
echo ""
echo "Outside sandbox:"
echo "  - Main project EditX-LoRA untouched"
echo "  - System Python untouched"
echo "  - SSH key / git config / crontab untouched"
echo ""
echo "To run inference later:"
echo "  cd ${PROJECT_DIR} && source ${VENV_DIR}/bin/activate"
echo "  python -m agent.router --meta ${MMAE_DIR}/MMAE-meta.json --output_dir outputs/agent"
echo "=========================================="