# AEC-2027-transfer-agent

> **NOT the official ICASSP 2027 Audio Editing Challenge submission.**
>
> This repository is a **backup / contrast / fail-safe** solution.
> The official entry is in the **Agent Track** under a different repository
> using **Step-Audio-EditX + LoRA** as the base model.
>
> This backup uses **microsoft/VibeVoice-ASR + openbmb/MiniCPM-o-2_6 + DSP
> tools** as the base, intentionally different to provide comparison
> data if the official entry fails reproducibility checks.

## What is this?

An Agent-based pipeline for the ICASSP 2027 Audio Editing Challenge (AEC).
Given an audio clip and a natural-language editing instruction in
English or Chinese, the agent picks a backend (text-conditioned
regeneration, volume / speed / pitch DSP, etc.) and produces an edited
.wav plus a JSON record.

Layout:

```
agent/router.py          -- Agent router (MMAE-format output)
agent/downloader.py      -- GitHub-first / HF-fallback model downloader
tools/setup_autodl.sh    -- Sandboxed deploy into /root/minimax on AutoDL
tools/make_fixture.py    -- Synthesized MMAE fixture for offline smoke
tools/run_agent.sh       -- One-shot inference runner
paper/paper.md           -- 2-page ICASSP-style paper draft
outputs/test_fixture/    -- Smoke-test output (5 samples, all routes OK)
SHA256SUMS.txt           -- SHA-256 of every committed file
MANIFEST.json            -- Same data as JSON for tooling
LICENSE                  -- MIT
```

## Why "backup"?

The author is also submitting an **Agent Track** entry built on
**Step-Audio-EditX + LoRA** under a separate repo. That submission is
the priority and is configured as the **official** entry for the
challenge. This repo is intentionally a different base (VibeVoice-ASR +
MiniCPM-o + DSP) so that:

1. If the official submission fails reproducibility, this repo provides
   an immediately-comparable second attempt without re-tooling.
2. The two approaches can be contrasted in the 2-page ICASSP paper.

## Reproducing locally

Requirements:
- Python 3.11
- A CUDA GPU (RTX 4090 / A100 recommended; smoke test runs on CPU)
- HF mirror endpoint reachable (set `HF_ENDPOINT=https://hf-mirror.com`)

```bash
git clone https://github.com/jxzhen/AEC-2027-transfer-agent
cd AEC-2027-transfer-agent
python3.11 -m venv .venv
source .venv/bin/activate          # Linux
.venv\Scripts\activate            # Windows
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "transformers>=4.51.3,<5.0.0" accelerate numpy scipy librosa soundfile \
    ml-collections absl-py huggingface_hub
HF_ENDPOINT=https://hf-mirror.com \
HF_HOME=./hf_cache \
python tools/make_fixture.py --out outputs/test_fixture --n 5
python -m agent.router \
    --meta outputs/test_fixture/MMAE-meta.json \
    --output_dir outputs/test_fixture
```

Output format follows the official MMAE submission spec:

```
predictions.json     -- full per-sample record
submission.jsonl     -- one JSON per line, for leaderboard ingestion
audio/<sample>.wav   -- edited audio
run_meta.json        -- tool versions, hashes, route counts
SHA256SUMS.txt       -- per-file checksums
```

## Reproducing on AutoDL Beijing B (sandboxed)

The author uses an AutoDL Beijing-B instance (da414aae78-...) and
deploys everything strictly under `/root/minimax`. The main project
EditX-LoRA is **never** read or modified; the script uses
runtime auto-discovery to blacklist every other folder under `/root/`
so the main project is guaranteed untouched.

```bash
# inside the AutoDL SSH session:
echo '<base64-of-tools/setup_autodl.sh>' | base64 -d > setup_hk_sandbox.sh
bash setup_hk_sandbox.sh
```

Or, equivalently:

```bash
cd ~
curl -L -o setup_hk_sandbox.sh \
    https://github.com/jxzhen/AEC-2027-transfer-agent/raw/main/tools/setup_autodl.sh
bash setup_hk_sandbox.sh
```

The script will:
1. Verify it is running on AutoDL (data disk detection)
2. Create `/root/autodl-tmp/minimax/` as the sandbox root
3. Install ffmpeg, git, venv, transformers, torch (CUDA or CPU)
4. Pull `microsoft/VibeVoice-ASR` and `openbpm/MiniCPM-o-2_6`
5. Run a 5-sample smoke test through the agent router

## How the Agent picks a backend

`agent/router.py` implements a rule-based router (no GPU needed for
classification). It emits one of these route tags per sample:

| Tag             | Backend                                    |
|-----------------|--------------------------------------------|
| `edit`          | MiniCPM-o + VibeVoice-ASR regenerate       |
| `volume`        | DSP gain in dB                             |
| `speed`         | DSP time-stretch                           |
| `dsp_pitch`     | DSP pitch-shift                            |
| `noise`         | DSP denoise / spectral gating              |
| `unrecognized`  | Fallback to passthrough + low-confidence   |

For the ICASSP submission the **edit** route is the only one that
counts; DSP backends are how the router avoids wasting GPU on
trivially-handleable instructions.

## Model downloader (GitHub-first with HF fallback)

`agent/downloader.py` implements the author-defined download rule:

> For audio / audio-editing models that exist on BOTH GitHub and
> HuggingFace, prefer GitHub (because the local Windows machine has
> FastGithub installed, so GitHub downloads are fast locally). Fall
> back to HuggingFace via the HK rented server only when (a) HF is
> newer than GitHub, or (b) HF-only.

Usage:

```bash
# Decide + download (sandbox venv active)
SANDBOX=/root/minimax HF_ENDPOINT=https://hf-mirror.com \
python -m agent.downloader \
    --hf microsoft/VibeVoice-ASR \
    --gh microsoft/VibeVoice \
    --out /root/minimax/hf_cache

# Decide only (no download, fast)
python -m agent.downloader \
    --hf openbmb/MiniCPM-o-2_6 \
    --gh OpenBMB/MiniCPM-o \
    --dry-run

# HF-only (no GitHub candidate)
python -m agent.downloader --hf some/repo --out /root/minimax/hf_cache
```

Decision logic:
1. No GitHub candidate -> HF.
2. GitHub repo has no release assets >= 100 MB -> HF.
3. Both GitHub (with weights) and HF have a timestamp:
   - If HF timestamp > GitHub timestamp -> HF.
   - Otherwise -> GitHub.
4. GitHub timestamp missing -> HF (defensive).
5. HF timestamp missing -> GitHub (defensive).

Sandbox safety: any output path that resolves outside the env-set
`SANDBOX` root is rejected with `[SANDBOX BLOCK]` and `exit 1`.

## Limitations

- The rule-based router is intentionally simple. For higher accuracy
  a fine-tuned LLM router (e.g. MiniCPM-o itself) could replace it.
- `microsoft/VibeVoice-ASR` and `openbmb/MiniCPM-o-2_6` were chosen
  for model-size diversity vs. the official submission's base.
- The paper draft (`paper/paper.md`) is a sketch, not a finished
  manuscript.

## License

MIT -- see `LICENSE`. Note the upstream model licenses apply separately.

## Contact

Jianxi Zheng -- hunan08182026@126.com