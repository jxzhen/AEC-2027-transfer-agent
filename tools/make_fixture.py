"""Generate a synthetic MMAE test fixture for agent smoke testing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)
    sr = 16000
    instructions = [
        "replace the background with silence",
        "make it louder",
        "speed up the recording 1.5x",
        "remove the noise from the audio",
        "shift pitch up by 2 semitones",
        "soften the speaker voice",
        "drop the volume by 3 dB",
        "increase the tempo",
    ]
    items = []
    for i in range(args.n):
        duration = 1.0 + (i % 3) * 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        f0 = 220.0 + 60.0 * np.sin(2 * np.pi * 0.5 * t)
        y = 0.5 * np.sin(2 * np.pi * f0 * t)
        y = y / max(np.abs(y).max(), 1e-6) * 0.8
        item_id = f"mmae_{i+1:03d}"
        wav_path = base / (item_id + ".wav")
        sf.write(wav_path, y.astype(np.float32), sr)
        items.append({
            "id": item_id,
            "audio": wav_path.name,
            "instruction": instructions[i % len(instructions)],
        })

    (base / "MMAE-meta.json").write_text(
        json.dumps({"data": items}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote synthetic MMAE fixture with {len(items)} items at {base}")


if __name__ == "__main__":
    main()