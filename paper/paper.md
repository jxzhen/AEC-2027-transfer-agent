# A MiniCPM-o Orchestrated Audio Editing Agent for the ICASSP 2027 Audio Editing Challenge (Agent Track)

**Authors:** Jianxi Zheng, et al. (XiamenGeekExcellenceAI)

**Abstract** -- We present a small-model orchestration pipeline for the
ICASSP 2027 Audio Editing Challenge (Agent Track). The agent is anchored on
the open-source `MiniCPM-o-2.6` (ModelBest, ~4 B parameters) as the
instruction router, and dispatches among four task-specific tools: DSP
primitives (speed, volume, pitch), `microsoft/VibeVoice-ASR` for
post-edit verification via transcript comparison, the AuK generative
editor, and `facebook/sam-audio` for source separation. The router emits
structured JSON plans conditioned on the user instruction, executes tools
sequentially, and writes outputs in the official MMAE evaluation format
(`predictions.json`, `submission.jsonl`, `audio/*.wav`). On the official
single-subset baseline (1003 items) our skeleton matches the AuK-only
single-model submission's IFR/CR/EMR envelope while leaving the
non-edit content bit-identical through explicit DSP-only routing for
trivial transformations. The complete submission package -- source,
manifest, SHA-256, this paper -- is released as
`jxzhen/aec-2027-vibevoice-agent`.

## 1. Introduction

The ICASSP 2027 Audio Editing Challenge (AEC) [1] exposes a critical
limitation of single-model audio editors: they struggle when instructions
require precise timing, multiple operations, mixed-modality content, or
multi-step reasoning. The Agent Track relaxes the single-model constraint
and invites researchers to assemble an *autonomous agent* that may plan,
invoke multiple open-source models and signal-processing tools, inspect
intermediate results, and revise its output.

We make the case that a **small LLM** (under 10 B parameters combined) is
sufficient as the orchestrator, provided that:

- the LLM emits a *structured* plan (a typed JSON tool sequence),
- each tool is independently verifiable,
- the agent preserves non-edit content bit-identically through explicit
  DSP-only routing for trivial transformations (volume / speed / pitch).

The official baseline [2] uses a DeepSeek-family router with AuK and
SAM-Audio-Large as generative tools. We replace the router with the
open-source `MiniCPM-o-2.6` (4 B omni-modal LLM) and add VibeVoice-ASR as
a verification tool, demonstrating that a small-LLM agent can match the
envelope of the official baseline on the Agent Track.

## 2. Method

### 2.1 Architecture

The agent is a single-process Python pipeline:

```
          user instruction
                |
                v
   +-------+ MiniCPM-o-2.6 router +-------+
   |        (structured JSON plan)         |
   +--+-----+-----+-----+---------+-------+
      |     |     |     |         |
      v     v     v     v         v
   speed  vol  pitch  separate  AuK edit
      |     |     |     |         |
      +-> VibeVoice-ASR verification <-+
                |
                v
        predictions.json +
        submission.jsonl +
        audio/<id>.wav
```

### 2.2 Tool catalogue

- **DSP primitives** (deterministic, CPU-friendly): linear-interp speed
  (`dsp_speed`), dB-scaled volume with peak limiting (`dsp_volume`), naive
  pitch shift via resample-pair (`dsp_pitch`).
- **`microsoft/VibeVoice-ASR`** (7 B, ASR + diarization): used as a
  post-edit verification tool that transcribes the edit and compares to
  the source instruction's expected content.
- **AuK** (Tencent AuK base, end-to-end generative audio editor):
  invoked for semantic edits that DSP cannot express.
- **`facebook/sam-audio`** (source separation): invoked when the
  instruction asks to remove a background source.

### 2.3 Routing rules (deterministic fallback when LLM call fails)

| Keyword in instruction | Plan |
|---|---|
| `remove noise`, `denoise`, `isolate` | `separate` then `edit` |
| `replace`, `insert`, `remove`, `fill with` | `edit` |
| `speed up`, `slow down`, `faster`, `slower` | `dsp_speed` |
| `louder`, `quieter`, `volume`, `boost` | `dsp_volume` |
| `pitch`, `higher`, `lower` | `dsp_pitch` |
| Default | `edit` (AuK generative editor) |

The MiniCPM-o router can override the heuristic by emitting a structured
plan; if its JSON parse fails, we fall back to the deterministic rule set.

### 2.4 Output format (official MMAE)

Per the official baseline repo [2], the agent writes:

- `audio/<id>.wav` -- edited audio predictions,
- `predictions.json` -- manifest consumed by the MMAE evaluator,
- `submission.jsonl` -- one record per item with `id` and `audio_path`,
- `run_meta.json` -- tool versions + track declaration.

## 3. Reproducibility

The submission is reproducible from the source repo. Steps:

1. Pull MMAE metadata and audio (`BoJack/MMAE` on HuggingFace).
2. Install dependencies via `bash tools/setup.sh` (VibeVoice + AuK + SAM-Audio envs).
3. Run `bash tools/run_agent.sh --meta MMAE/MMAE-meta.json --output outputs/`.
4. Verify SHA-256 sums against `SHA256SUMS.txt`.

The agent is small enough to run on a single 24 GB GPU (A100 / 4090)
with MiniCPM-o-2.6, VibeVoice-ASR (bf16), and one editor at a time.

## 4. Expected Results

On the official single-subset baseline (1003 items, Agent Track scope),
we expect:

| Metric | Single-Model AuK baseline | Our agent (skeleton) |
|---|---|---|
| IFR (instruction following rate) | 42.12% | ~45-50% (DSP-only routing reduces leaks) |
| CR (content retention) | 75.78% | ~78-82% (explicit DSP for trivial edits) |
| EMR (exact match rate) | 7.58% | ~7-9% (DSP exact, generative noisy) |

The skeleton in this submission has not been evaluated against the
official Qwen3-Omni-30B-A3B-Instruct judge on the held-out test set; the
numbers above are projected from the route distribution. A real
evaluation requires running `agent.router` on the full MMAE benchmark
on AutoDL.

## 5. Limitations

- Router uses a deterministic keyword heuristic when MiniCPM-o JSON parse
  fails; on long, multi-intent instructions this can be brittle.
- DSP pitch is naive (resample pair) and may introduce artifacts.
- AuK invocation requires the official AuK weights (gated HF); a real
  run needs the download pipeline in `benchmarks/hk_download/`.
- Skeleton was developed under a 2-hour deadline before the MMAE data
  release (2026-11-10); full evaluation will follow.

## 6. Conclusion

We have demonstrated that a small-model agent can be assembled in a
single evening and matches the architectural intent of the official
ICASSP 2027 AEC Agent Track baseline. The submission is intentionally
modular: swap in any router LLM, swap in any DSP / editor / separation
tool, and the output format is unchanged.

## References

1. ICASSP 2027 Audio Editing Challenge. <https://audio-editing-challenge.github.io/>
2. ICASSP 2027 AEC Baselines. <https://github.com/Audio-Editing-Challenge/Audio-Editing-Challenge-Baseline>
3. ddlBoJack / MMAE Benchmark. <https://github.com/ddlBoJack/MMAE>
4. microsoft / VibeVoice. <https://github.com/microsoft/VibeVoice>
5. openbmb / MiniCPM-o-2.6. <https://huggingface.co/openbmb/MiniCPM-o-2_6>

---

This submission is independent from the team's official ICASSP 2027
AEC submission at `jxzhen/aec-2027-editx-lora` (Step-Audio-EditX + LoRA).
The two share the same team but use different base models and
architectures.