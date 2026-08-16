"""Build a release ZIP from the current git commit (source only, no deps).

Usage:
    .venv/bin/python scripts/release.py "commit message"      # commit + zip
    .venv/bin/python scripts/release.py "commit message" --push   # also git push

Commits any pending changes, then produces ``entropia-riko-release.zip`` in the
project's parent directory (outside the repo, so the working folder is never
touched). Only tracked files are included (``.gitignore`` excludes deps/caches).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "entropia-riko-release.zip"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, check=check, capture_output=True, text=True)


def main() -> None:
    argv = sys.argv[1:]
    do_push = "--push" in argv
    msg = " ".join(a for a in argv if a != "--push") or "Release"

    run("git", "add", "-A")
    status = run("git", "status", "--porcelain").stdout.strip()
    if status:
        run("git", "commit", "-q", "-m", msg)

    run("git", "archive", "--format=zip", "-o", str(OUT), "HEAD")

    if do_push:
        r = run("git", "push", check=False)
        if r.returncode != 0:
            print("push 失败（可能还没配置 remote）：")
            print(r.stderr.strip())

    h = run("git", "rev-parse", "--short", "HEAD").stdout.strip()
    print(f"release: {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, git {h})")


if __name__ == "__main__":
    main()
