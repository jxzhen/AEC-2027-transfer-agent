#!/usr/bin/env python3
"""
GitHub-first model downloader with HF-via-HK fallback.

Rule (per user-defined preference, 2026-09-30):
  1. For audio / audio-editing models that exist on BOTH GitHub and HF,
     prefer GitHub because the local Windows machine has FastGithub
     (faster than HF, which has no local proxy).
  2. Fall back to HF (via the HK rented server) only when:
     a) HF version is NEWER than GitHub release, OR
     b) The model / weights exist on HF but NOT on GitHub at all.

Sandbox constraint:
  All output paths MUST be inside the sandbox (/root/minimax or
  /root/autodl-tmp/minimax on AutoDL). The script asserts this and
  refuses to write outside.

Usage:
  # Decide and download (inside sandbox venv)
  python -m agent.downloader \
      --hf microsoft/VibeVoice-ASR \
      --gh microsoft/VibeVoice \
      --out /root/minimax/hf_cache

  # Decide only (no download)
  python -m agent.downloader \
      --hf openbmb/MiniCPM-o-2_6 \
      --gh OpenBMB/MiniCPM-o \
      --dry-run

  # HF-only (no GitHub candidate)
  python -m agent.downloader --hf some/repo --out /root/minimax/hf_cache
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# gh-proxy.com mirrors github.com (China-friendly, used by AutoDL/HK)
GH_PROXY = "https://gh-proxy.com/"
HF_MIRROR = "https://hf-mirror.com"

# Sandbox root (will be overridden by env var SANDBOX in sandboxed runs)
SANDBOX_DEFAULT = "/root/minimax"

# Minimum weight-file size to count as "real weights" (100 MB)
MIN_WEIGHT_BYTES = 100 * 1024 * 1024

# Known audio / audio-editing models: HF repo -> (GitHub repo, weights_on_github)
KNOWN_MODELS = {
    "microsoft/VibeVoice-ASR":  ("microsoft/VibeVoice", False),
    "openbmb/MiniCPM-o-2_6":    ("OpenBMB/MiniCPM-o",   False),
    "openbmb/MiniCPM-o-2_6-int4": ("OpenBMB/MiniCPM-o", False),
}


# ---------------------------------------------------------------------------
# Sandbox safety
# ---------------------------------------------------------------------------
def assert_inside_sandbox(path: str) -> None:
    """Refuse to write or read any path outside the sandbox."""
    sandbox = os.environ.get("SANDBOX", SANDBOX_DEFAULT)
    # Normalize both paths to forward-slash form for cross-platform comparison
    sandbox_n = sandbox.replace("\\", "/").rstrip("/")
    if os.path.isabs(path):
        real = os.path.abspath(path).replace("\\", "/")
    else:
        real = path.replace("\\", "/")
    if not (real == sandbox_n or real.startswith(sandbox_n + "/")):
        print(f"[SANDBOX BLOCK] {path} -> {real} not under {sandbox_n}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# GitHub inspection
# ---------------------------------------------------------------------------
def _gh_api(url: str, timeout: int = 10) -> Optional[dict]:
    """GET a GitHub API endpoint, return parsed JSON or None."""
    try:
        r = subprocess.run(
            ["curl", "-s", "-L", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return json.loads(r.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def github_has_weights(repo_id: str) -> Tuple[bool, list]:
    """Return (has_weights, list_of_asset_urls)."""
    # 1. Check the latest release's assets
    data = _gh_api(f"https://api.github.com/repos/{repo_id}/releases/latest")
    assets = []
    if data and isinstance(data.get("assets"), list):
        for a in data["assets"]:
            if a.get("size", 0) >= MIN_WEIGHT_BYTES:
                assets.append(a.get("browser_download_url"))
    # 2. If no release assets, walk the repo tree for large files via LFS
    if not assets:
        data = _gh_api(f"https://api.github.com/repos/{repo_id}/contents/")
        # This is a shallow check; for production we'd recurse
        if isinstance(data, list):
            for entry in data:
                if entry.get("type") == "dir" and entry.get("name", "").lower() in {
                    "weights", "checkpoints", "model", "models",
                }:
                    assets.append(f"github://{repo_id}/tree/{entry.get('name')}")
    return (len(assets) > 0, assets)


def github_latest_release_time(repo_id: str) -> Optional[datetime]:
    """Latest GitHub release published_at (timezone-aware) or None."""
    data = _gh_api(f"https://api.github.com/repos/{repo_id}/releases/latest")
    if not data or "published_at" not in data:
        return None
    try:
        ts = data["published_at"].rstrip("Z")
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def github_repo_pushed_time(repo_id: str) -> Optional[datetime]:
    """Fallback if no release: repo's last push timestamp."""
    data = _gh_api(f"https://api.github.com/repos/{repo_id}")
    if not data or "pushed_at" not in data:
        return None
    try:
        ts = data["pushed_at"].rstrip("Z")
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HuggingFace inspection
# ---------------------------------------------------------------------------
def hf_latest_modified(repo_id: str) -> Optional[datetime]:
    """Latest commit timestamp of a HF model repo, or None."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.repo_info(repo_id, repo_type="model")
        return info.last_modified
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Decision logic (the user-defined rule)
# ---------------------------------------------------------------------------
def decide_source(repo_hf: str, repo_gh: Optional[str]) -> Tuple[str, str]:
    """Return (source, reason) where source is 'github' or 'hf'."""
    if not repo_gh:
        return ("hf", f"No GitHub candidate provided; using HF: {repo_hf}")

    has_weights, _ = github_has_weights(repo_gh)
    if not has_weights:
        return ("hf", f"GitHub {repo_gh} has no model weights; using HF: {repo_hf}")

    # Both GitHub (with weights) and HF exist -> compare recency
    gh_time = github_latest_release_time(repo_gh) or github_repo_pushed_time(repo_gh)
    hf_time = hf_latest_modified(repo_hf)

    if gh_time and hf_time:
        if hf_time > gh_time:
            return ("hf", f"HF newer than GitHub ({hf_time} > {gh_time}); using HF: {repo_hf}")
        return ("github", f"GitHub {repo_gh} up-to-date ({gh_time} >= {hf_time}); using GitHub")

    if hf_time and not gh_time:
        return ("hf", f"GitHub {repo_gh} has no release timestamp; using HF: {repo_hf}")

    return ("github", f"GitHub {repo_gh} has weights and HF timestamp unavailable; using GitHub")


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------
def download_from_github(repo_gh: str, asset_urls: list, out_dir: str) -> list:
    """Download GitHub release assets via gh-proxy.com."""
    assert_inside_sandbox(out_dir)
    out = Path(out_dir) / ("github_" + repo_gh.replace("/", "_"))
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in asset_urls:
        if url.startswith("github://"):
            print(f"[github] skipping non-downloadable tree: {url}")
            continue
        proxy_url = GH_PROXY + url
        fname = Path(url).name
        dst = out / fname
        print(f"[github] {url} -> {dst}")
        print(f"[github] via proxy {proxy_url}")
        subprocess.run(["curl", "-L", "--fail", "-o", str(dst), proxy_url], check=False)
        paths.append(str(dst))
    return paths


def download_from_hf(repo_hf: str, out_dir: str) -> str:
    """Download weights from HF via hf-mirror.com."""
    assert_inside_sandbox(out_dir)
    cache = out_dir
    os.makedirs(cache, exist_ok=True)
    # Sandbox re-assertion
    real_cache = os.path.abspath(cache)
    sandbox = os.environ.get("SANDBOX", SANDBOX_DEFAULT)
    if not (real_cache == sandbox or real_cache.startswith(sandbox + "/")):
        print(f"[SANDBOX BLOCK] HF cache escapes sandbox: {real_cache}", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import snapshot_download
    print(f"[hf] downloading {repo_hf} -> {cache}")
    return snapshot_download(
        repo_id=repo_hf,
        cache_dir=cache,
        allow_patterns=[
            "*.json", "*.txt", "*.md",
            "*.safetensors", "*.bin", "*.pt",
            "tokenizer.*", "*.py",
        ],
        max_workers=8,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="GitHub-first model downloader with HF-via-HK fallback")
    p.add_argument("--hf", required=True, help="HuggingFace repo_id, e.g. microsoft/VibeVoice-ASR")
    p.add_argument("--gh", default=None, help="Optional GitHub repo_id, e.g. microsoft/VibeVoice")
    p.add_argument("--out", default=os.path.join(SANDBOX_DEFAULT, "hf_cache"),
                   help="Output dir (MUST be inside the sandbox)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print decision + reason only, do not download")
    p.add_argument("--endpoint", default=HF_MIRROR,
                   help=f"HF endpoint (default {HF_MIRROR})")
    args = p.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint

    source, reason = decide_source(args.hf, args.gh)
    print(f"[decision] {source.upper()}: {reason}")

    if args.dry_run:
        return 0

    if source == "github":
        has, urls = github_has_weights(args.gh)
        if has:
            paths = download_from_github(args.gh, urls, args.out)
            for p in paths:
                print(f"[ok] {p}")
        else:
            # Defensive fallback: GitHub reported no weights, switch to HF
            print("[fallback] GitHub has no weights; switching to HF")
            p = download_from_hf(args.hf, args.out)
            print(f"[ok] {p}")
    else:
        p = download_from_hf(args.hf, args.out)
        print(f"[ok] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())