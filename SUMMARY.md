# ICASSP 2027 Audio Editing Challenge -- Final Summary

**Date**: 2026-09-30
**Author**: Jianxi Zheng (hunan08182026@126.com)
**Sandbox deadline**: 21:30 (closed)
**Official ICASSP deadline**: 2026-11-25 23:59 PST

---

## 1. What this is

Three **backup / contrast / fail-safe** GitHub repos for the ICASSP 2027
Audio Editing Challenge (AEC). These repos are **NOT the official
submission**.

The official entry is `jxzhen/aec-2027-editx-lora`
(Step-Audio-EditX + LoRA, Agent Track). The three backup repos below
provide a different base (VibeVoice-ASR + MiniCPM-o-2_6 + DSP tools)
as a comparison / contrast / fail-safe artifact.

---

## 2. Repositories pushed to GitHub

| # | URL | Role | Files | Public |
|---|---|---|---|---|
| 1 | https://github.com/jxzhen/AEC-2027-transfer-agent | Agent-based backup solution (MMAE format) | 35 | yes |
| 2 | https://github.com/jxzhen/vibevoice-based_weight | Weight pointer + HK download + ASR/AEC benchmarks | 19 | yes |
| 3 | https://github.com/jxzhen/vibevoice-based_AudioEditing.html | Local HTML dashboard for experiment comparison | 8 | yes |

All three are public, ASCII-clean, MIT-licensed, and clearly marked
"NOT the official ICASSP submission".

---

## 3. Sandbox deployment

A hardened sandbox deploy script is committed to repo 1:

```
tools/setup_autodl.sh  (9396 bytes, 0 non-ASCII, 3 layers of protection)
```

- L1 path whitelist (`/root/minimax` only)
- L2 explicit blacklist (30+ main-project-related names)
- L3 runtime auto-discovery of every `/root/*/` directory
- Auto-relocate to `/root/autodl-tmp/minimax` if data disk exists
- Skips auto-discovered folders that are ancestors of the sandbox

Deploy on AutoDL Beijing B (`da414aae78-60035a77`, SSH
`-p 32033 root@connect.bjb1.seetacloud.com`):

```bash
curl -L -o setup_autodl.sh \
    https://github.com/jxzhen/AEC-2027-transfer-agent/raw/main/tools/setup_autodl.sh
bash setup_autodl.sh
```

---

## 4. GPU inference (after AutoDL is upgraded to RTX 4090)

A single-command runner is committed to repo 1:

```
tools/run_full_inference.sh
```

```bash
cd /root/autodl-tmp/minimax/aec-2027-transfer-agent
bash tools/run_full_inference.sh
```

This will:
1. Detect GPU, reinstall CUDA PyTorch (cu121)
2. Download `BoJack/MMAE` dataset (HF mirror)
3. Run `agent.router` on the full MMAE set
4. Emit `predictions.json` + `submission.jsonl` + `audio/*.wav` +
   `traces/*.json` + `run_meta.json` + `SHA256SUMS.txt`

---

## 5. Sandbox isolation guarantees

The following were verified untouched at all times:
- `jxzhen/aec-2027-editx-lora` (main project) -- never read or modified
- SSH key on HK / AutoDL -- never touched
- `git config` on HK / AutoDL -- never modified
- `~/.bashrc`, crontab -- never modified
- `/root/editx-lora`, `/root/EditX`, etc. -- explicitly blacklisted

---

## 6. Known limitations / TODO

- Real GPU inference not yet executed (AutoDL instance is CPU-only;
  user needs to upgrade to RTX 4090 in the AutoDL web console)
- The 5-sample smoke test on the local Windows machine validates the
  router logic (MMAE-format output, all 5 routes correct) but does NOT
  exercise VibeVoice-ASR / MiniCPM-o inference (those models were not
  downloaded to the local machine, only to AutoDL CPU sandbox)
- The 2-page ICASSP paper draft (`paper/paper.md`) is a sketch

---

## 7. Contact

Jianxi Zheng -- hunan08182026@126.com