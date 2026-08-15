# Cross-Platform Standard

Entropia Riko must run identically on **macOS**, **Windows**, and **Linux** —
both as a browser web app and as a standalone Electron desktop app. Execution is
**CPU-first**, with CUDA (NVIDIA) and MPS (Apple Silicon) detected automatically
when available. macOS is the current development/test environment, but nothing
in the codebase may assume it.

---

## Delivery: browser vs Electron desktop

The same frontend and backend serve two shells — users choose one:

### Browser (web app)

```text
Vite dev server   http://localhost:5173   (React UI)
FastAPI           http://localhost:8000   (Python runtime)
```

The Vite dev server proxies `/api` requests to `:8000`, so the UI talks to the
Python runtime over HTTP exactly as it would in any environment. Two terminals:

```bash
# Terminal 1 — API
.venv/bin/python -m uvicorn src.server.app:app --reload --port 8000

# Terminal 2 — frontend
npm run dev
```

### Electron desktop (`npm run desktop`)

`electron/main.js` + `electron/preload.js` wrap the same web app in a native
window:

- On launch it **spawns the FastAPI backend** from the project `.venv`, then
  opens a `BrowserWindow` (1440×900, min 900×600) pointed at the Vite dev server
  (`http://localhost:5173`, overridable via the `RIKO_DEV_URL` env var).
- `preload.js` exposes nothing: `contextIsolation` is on, `nodeIntegration` is
  off, and the UI still talks to FastAPI over HTTP — no platform-specific IPC
  exists.
- External links open in the system browser (`setWindowOpenHandler` →
  `shell.openExternal`).
- On `window-all-closed` the shell kills the spawned API process and quits —
  except on macOS, where it stays resident and recreates the window on
  `activate` (standard macOS convention).

Startup:

```bash
npm install --save-dev electron
npm run dev &            # keep the Vite dev server running
npm run desktop
```

---

## Python environment (venv)

A virtual environment is the only supported way to run the backend:

```bash
python -m venv .venv

# activate — POSIX (macOS / Linux):
. .venv/bin/activate

# activate — Windows (PowerShell):
.venv\Scripts\Activate.ps1

# or Windows (cmd):
.venv\Scripts\activate.bat

pip install -r requirements.txt
```

- The **venv path is platform-aware**. The Electron shell selects it explicitly
  rather than relying on an activated shell:

```js
const python = process.platform === "win32"
  ? path.join(root, ".venv", "Scripts", "python.exe")
  : path.join(root, ".venv", "bin", "python");
```

- If the venv is missing, the desktop shell falls back to assuming the API is
  already running on `:8000`; it does not crash.

---

## Path handling rules

- Use `pathlib.Path` for **all** Python filesystem work — never string
  concatenation of separators.
- Do **not** hardcode `/Users/...` or `C:\...`. Resolve user-supplied paths with
  `Path(p).expanduser().resolve()` (the `~` form works on all three OSes).
- The project root is derived from the source file's location, not the CWD:

```python
# src/runtime/subgraph.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

- Normalize Windows separators at the boundary. Both `_resolve_working` (the
  project file manager) and `resolve_graph_file` (subgraph resolution) do
  `s.replace("\\", "/")` before matching, so a `C:\...` or `\`-separated input
  still behaves.
- Guard filesystem endpoints: the built-in `.riko`/`.ric` reader accepts only
  files whose suffix is `.riko`/`.ric` **and** whose resolved path is inside
  `PROJECT_ROOT` (`_safe_project_file`); the working-folder manager refuses paths
  that escape the working root (`_resolve_working`).
- The built-in file explorer uses `Path(path or "~").expanduser().resolve()` and
  supports `~`, `~/Desktop`, `~/Documents`, `~/Downloads`, plus a "Working
  Folder" quick-access entry.

---

## Device rules (runtime)

- **CPU execution must always work** on every platform — no GPU required.
- CUDA and MPS are used only after availability checks; `resolve_device("auto")`
  falls back CUDA → MPS → CPU and never raises (see `TORCH_BACKEND.md`).
- Requesting an unavailable accelerator explicitly (e.g. `"cuda"` on a
  CPU-only machine) raises a clear error instead of silently degrading.

---

## Text & encoding

- All text files read/written by the app use **UTF-8** (`encoding="utf-8"`).
- JSON serialization uses `ensure_ascii=False` so non-ASCII labels round-trip
  on all platforms.

---

## General rules

- Keep platform-neutral logic (tensor IR, document model, executor) free of OS
  imports and free of shell-command reliance — `src/core/` imports neither torch
  nor any platform module.
- Put platform-specific behavior behind small adapters or fallback logic
  (e.g. the venv path selection above) rather than sprinkling `if` checks
  through core logic.
- The frontend is plain web tech (React + Vite) and is inherently portable; it
  must not assume a browser-only or Electron-only environment (use the HTTP API
  for all backend access, never Electron-only IPC).
