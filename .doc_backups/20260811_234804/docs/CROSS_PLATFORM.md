# Cross-Platform Standard

## Purpose

Torch Houdini Node must be compatible with Windows, macOS, and Linux.

The primary test environment is currently macOS, but implementation decisions must not assume macOS-only behavior.

## Supported Platforms

The project should support:

- macOS
- Windows
- Linux

The first active development and testing environment is macOS.

## Core Rule

Do not write platform-specific code unless there is a clear adapter layer or fallback path.

The core graph, node system, tensor abstraction, and Torch backend should work the same way across platforms.

## Path Handling

Use cross-platform path APIs.

In Python:

- Prefer `pathlib.Path`.
- Avoid hardcoded `/` path separators.
- Avoid shell-specific path assumptions.
- Do not hardcode macOS paths such as `/Users/...`.
- Do not hardcode Windows paths such as `C:\...`.

Good:

```python
from pathlib import Path

config_path = Path("configs") / "default.yaml"
```

Avoid:

```python
config_path = "configs/default.yaml"
```

## Shell Commands

Do not rely on shell commands for core functionality.

Avoid requiring:

- `bash`
- `zsh`
- `tree`
- `grep`
- macOS-only commands
- Linux-only package tools
- Windows-only PowerShell behavior

If shell commands are used in documentation, provide platform-neutral alternatives when possible.

## Dependencies

Dependencies should be available on Windows, macOS, and Linux.

Before adding a dependency, check:

- Does it support all target platforms?
- Does it require compiled system libraries?
- Does it work on Apple Silicon?
- Does it work without Houdini installed?
- Does it work in CI?

## Houdini Integration

Houdini-specific code must be isolated.

The project should allow:

- Core tests to run without Houdini.
- Torch backend tests to run without Houdini.
- Houdini adapters to be skipped when Houdini Python APIs are unavailable.

Do not import Houdini modules in core code.

## PyTorch Device Handling

Device selection must be safe across platforms.

Supported device values:

- `cpu`
- `cuda`
- `mps`
- `auto`

Rules:

- CPU must always work.
- CUDA must only be used when available.
- MPS must only be used on supported Apple systems.
- Auto mode must fall back safely to CPU.

## File Encoding

Use UTF-8 for project files.

Avoid platform-dependent default encodings.

In Python file operations, specify encoding when reading or writing text:

```python
path.read_text(encoding="utf-8")
path.write_text(content, encoding="utf-8")
```

## Line Endings

Use LF line endings in the repository.

Tools should not depend on CRLF or LF behavior.

## Tests

Tests should be written so they can run on all platforms.

The active local test environment is macOS.

Recommended test strategy:

- Run local tests on macOS during development.
- Keep tests free of macOS-only paths and commands.
- Mark Houdini-dependent tests separately.
- Add CI later for Windows, macOS, and Linux.

## Agent Instructions

Any AI coding agent must follow these rules:

1. Assume the project must run on Windows, macOS, and Linux.
2. Treat macOS as the current test machine, not the only target platform.
3. Use cross-platform file and path APIs.
4. Keep shell commands out of core logic.
5. Keep Houdini integration optional.
6. Make CPU execution the guaranteed baseline.
7. Document any platform-specific limitation before implementing it.

## First Implementation Requirement

Stage 1 core graph code must be fully platform-neutral.

It must not depend on:

- Houdini.
- macOS-only paths.
- shell-specific behavior.
- GPU availability.
- system-level package managers.
