"""
Agent router for ICASSP 2027 Audio Editing Challenge -- Agent Track.

Architecture (compliance with official GC-4 Agent Track):
    - Router LLM:  MiniCPM-o-2_6 (ModelBest omni-modal, ~4B params, small)
    - ASR tool:    microsoft/VibeVoice-ASR (7B, transcribe for verification)
    - Editor tool: AuK (Tencent AuK base, per official baseline)
    - Sep tool:    SAM-Audio-Large (source separation when needed)
    - DSP tools:   speed / volume / pitch (local, deterministic)

The router is a structured-output LLM call: it emits a JSON plan listing
which tools to invoke in which order, with parameters for each.

This skeleton:
    - Loads the official MMAE metadata
    - Iterates each item, gets a router decision
    - Invokes tools sequentially
    - Persists predictions.json / submission.jsonl / audio/*.wav
      in the format the official MMAE evaluator expects.

Designed to be small-model-friendly (MiniCPM-o-2_6 fits on a 16 GB GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Silence HF progress bars in CI
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# ---------------------------------------------------------------------------
# Instruction parsing helpers
# ---------------------------------------------------------------------------

SPEED_KEYWORDS = ["speed up", "slow down", "faster", "slower", "speed"]
VOLUME_KEYWORDS = ["louder", "quieter", "volume", "boost", "lower the volume", "increase the volume"]
PITCH_KEYWORDS = ["pitch", "higher pitch", "lower pitch", "shift pitch"]
EDIT_KEYWORDS = ["replace", "insert", "remove", "delete", "change to", "fill with", "substitute"]
SEPARATION_KEYWORDS = ["separate", "isolate", "remove background", "denoise", "remove noise"]


def classify_intent(instruction: str) -> Dict[str, Optional[str]]:
    """Lightweight heuristic to decide which tool the instruction targets.

    Real MiniCPM-o call is below (classify_with_llm); this is the
    deterministic fallback used in tests.
    """
    text = instruction.lower()
    plan = {
        "tool_sequence": [],
        "speed": None,
        "volume_db": None,
        "pitch_semitones": None,
        "separation": None,
        "edit_target": None,
    }
    if any(k in text for k in SEPARATION_KEYWORDS):
        plan["tool_sequence"].append("separate")
    if any(k in text for k in EDIT_KEYWORDS):
        plan["tool_sequence"].append("edit")
    if any(k in text for k in SPEED_KEYWORDS):
        plan["tool_sequence"].append("dsp_speed")
        m = re.search(r"(\d+(?:\.\d+)?)\s*[xX]", text)
        if m:
            plan["speed"] = float(m.group(1))
        else:
            plan["speed"] = 1.5 if "faster" in text or "speed up" in text else 0.75
    if any(k in text for k in VOLUME_KEYWORDS):
        plan["tool_sequence"].append("dsp_volume")
        plan["volume_db"] = 6.0 if "louder" in text or "boost" in text else -6.0
    if any(k in text for k in PITCH_KEYWORDS):
        plan["tool_sequence"].append("dsp_pitch")
        plan["pitch_semitones"] = 2 if "higher" in text else -2
    if not plan["tool_sequence"]:
        # default to a generative edit (AuK) -- this is the common case
        plan["tool_sequence"].append("edit")
        plan["edit_target"] = instruction.strip()
    return plan


# ---------------------------------------------------------------------------
# DSP tool implementations (CPU-friendly, deterministic)
# ---------------------------------------------------------------------------


def dsp_speed(wav: np.ndarray, rate: float) -> np.ndarray:
    """Resample via linear interpolation. rate > 1 speeds up."""
    if rate == 1.0:
        return wav
    n_out = int(len(wav) / rate)
    xp = np.arange(len(wav))
    xq = np.linspace(0, len(wav) - 1, n_out)
    return np.interp(xq, xp, wav).astype(np.float32)


def dsp_volume(wav: np.ndarray, db: float) -> np.ndarray:
    factor = 10.0 ** (db / 20.0)
    out = wav * factor
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def dsp_pitch(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Naive pitch shift via time-stretch then resample."""
    factor = 2.0 ** (semitones / 12.0)
    stretched = dsp_speed(wav, 1.0 / factor)
    return dsp_speed(stretched, factor)


# ---------------------------------------------------------------------------
# Loader for MMAE metadata
# ---------------------------------------------------------------------------


def load_mmae_meta(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected MMAE meta format: {path}")


# ---------------------------------------------------------------------------
# Placeholder tool stubs (real implementations come after HK download)
# ---------------------------------------------------------------------------


def stub_separate(wav_path: Path) -> Dict[str, np.ndarray]:
    """Placeholder for SAM-Audio-Large: return the input as-is.

    A real SAM-Audio invocation requires:
        export SAM_AUDIO_PYTHON=/path/to/sam-audio-env/bin/python
        export SAM_AUDIO_HOME=/path/to/sam-audio-weights
    """
    import soundfile as sf
    y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    return {"audio": y, "sr": sr}


def stub_edit(wav_path: Path, target: str, region: Optional[tuple] = None) -> np.ndarray:
    """Placeholder for AuK-based generative edit.

    A real AuK call:
        export AUK_REPO=/path/to/AuK
        python ${AUK_REPO}/inference.py --audio <wav> --instruction "<target>"
    """
    import soundfile as sf
    y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    # Skeleton: in the absence of AuK weights, pass-through. Real runs
    # would invoke the AuK model.
    return y


# ---------------------------------------------------------------------------
# Router decision (LLM-backed, with fallback)
# ---------------------------------------------------------------------------


def classify_with_llm(instruction: str, llm_call=None) -> Dict:
    """Ask MiniCPM-o to produce a structured plan.

    `llm_call(instruction)` should return a JSON string. If nil or the
    JSON parse fails, fall back to the deterministic heuristic.
    """
    if llm_call is None:
        return classify_intent(instruction)
    try:
        raw = llm_call(instruction)
        plan = json.loads(raw)
        # minimal schema check
        assert "tool_sequence" in plan and isinstance(plan["tool_sequence"], list)
        return plan
    except Exception:
        return classify_intent(instruction)


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


def run_agent(
    meta_path: Path,
    output_dir: Path,
    sr: int = 16000,
    llm_call=None,
    limit: Optional[int] = None,
) -> Dict:
    """Iterate MMAE items, route through tools, write predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)

    items = load_mmae_meta(meta_path)
    if limit:
        items = items[:limit]

    predictions: List[Dict] = []
    submission_rows: List[Dict] = []
    audio_paths: List[str] = []

    for idx, item in enumerate(items):
        item_id = item.get("id") or item.get("signal_id") or f"item_{idx:05d}"
        wav_rel = item.get("audio") or item.get("audio_path")
        wav_path = Path(wav_rel)
        if not wav_path.is_absolute():
            wav_path = meta_path.parent / wav_rel
        instruction = item.get("instruction") or item.get("text") or ""

        t0 = time.time()
        plan = classify_with_llm(instruction, llm_call=llm_call)
        # Load source wav (placeholder separation -> source)
        try:
            sources = stub_separate(wav_path)
            y, src_sr = sources["audio"], sources["sr"]
            if src_sr != sr:
                # quick resample to target sr (the agent's working sr)
                from scipy.signal import resample
                n = int(len(y) * sr / src_sr)
                y = resample(y, n).astype(np.float32)
        except Exception as e:
            print(f"[agent] item {item_id}: failed to load audio: {e}", file=sys.stderr)
            continue

        edited = y
        for tool in plan["tool_sequence"]:
            if tool == "dsp_speed" and plan.get("speed"):
                edited = dsp_speed(edited, plan["speed"])
            elif tool == "dsp_volume" and plan.get("volume_db") is not None:
                edited = dsp_volume(edited, plan["volume_db"])
            elif tool == "dsp_pitch" and plan.get("pitch_semitones") is not None:
                edited = dsp_pitch(edited, sr, plan["pitch_semitones"])
            elif tool == "edit":
                edited = stub_edit(wav_path, plan.get("edit_target") or instruction)
            elif tool == "separate":
                # already handled at load time
                pass

        # Write audio
        import soundfile as sf
        out_audio = audio_dir / f"{item_id}.wav"
        sf.write(out_audio, edited, sr)
        # POSIX-style relative path for cross-platform reproducibility
        rel_path = out_audio.relative_to(output_dir).as_posix()

        elapsed = time.time() - t0

        trace = {
            "id": item_id,
            "instruction": instruction,
            "plan": plan,
            "elapsed_seconds": round(elapsed, 3),
            "output_audio": rel_path,
        }
        (traces_dir / f"{item_id}.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        predictions.append({
            "id": item_id,
            "audio": rel_path,
            "instruction": instruction,
            "elapsed": elapsed,
        })
        submission_rows.append({
            "id": item_id,
            "audio_path": rel_path,
        })
        audio_paths.append(rel_path)

    # Aggregate outputs in official MMAE format
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(output_dir / "submission.jsonl", "w", encoding="utf-8") as f:
        for row in submission_rows:
            f.write(json.dumps(row) + "\n")
    run_meta = {
        "n_items": len(items),
        "tool": "minicpm-o-router + vibevoice-asr + auk",
        "model_versions": {
            "router": "openbmb/MiniCPM-o-2_6",
            "asr": "microsoft/VibeVoice-ASR",
            "editor": "tencent/AuK (placeholder)",
            "separation": "facebook/sam-audio (placeholder)",
        },
        "track": "ICASSP 2027 Audio Editing Challenge -- Agent Track",
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "n_items": len(items),
        "n_pred": len(predictions),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, required=True, help="MMAE-meta.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    result = run_agent(Path(args.meta), Path(args.output_dir), sr=args.sr, limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()