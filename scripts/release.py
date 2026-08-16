"""Build a release ZIP and optionally publish it to GitHub.

Usage:
    .venv/bin/python scripts/release.py "commit message"
        commit pending changes + build entropia-riko-release.zip (parent dir)

    .venv/bin/python scripts/release.py "message" --push
        also git push

    .venv/bin/python scripts/release.py "message" --release v0.1.0
        also publish the ZIP as a GitHub Release asset (needs `gh auth login`)

    .venv/bin/python scripts/release.py "message" --release v0.1.0 --push
        everything: commit + zip + GitHub Release + git push
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
    release_tag: str | None = None
    words: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--release" and i + 1 < len(argv):
            release_tag = argv[i + 1]
            i += 2
            continue
        if a == "--push":
            i += 1
            continue
        words.append(a)
        i += 1
    msg = " ".join(words) or "Release"

    run("git", "add", "-A")
    if run("git", "status", "--porcelain").stdout.strip():
        run("git", "commit", "-q", "-m", msg)

    run("git", "archive", "--format=zip", "-o", str(OUT), "HEAD")

    if release_tag:
        r = run(
            "gh", "release", "create", release_tag, str(OUT),
            "--title", f"Entropia Riko {release_tag}",
            "--notes", msg,
            check=False,
        )
        if r.returncode != 0:
            print("gh release 失败（确认已 `gh auth login` 且 tag 未被占用）：")
            print(r.stderr.strip())

    if do_push:
        r = run("git", "push", check=False)
        if r.returncode != 0:
            print("push 失败（确认已配置 remote）：")
            print(r.stderr.strip())

    h = run("git", "rev-parse", "--short", "HEAD").stdout.strip()
    print(f"release: {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, git {h})")


if __name__ == "__main__":
    main()
