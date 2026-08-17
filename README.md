# Entropia Riko

[中文](README_CN.md) | English

[![PyPI version](https://img.shields.io/pypi/v/entropia-riko.svg)](https://pypi.org/project/entropia-riko/)
[![GitHub release](https://img.shields.io/github/v/release/SakuraEntropia/Entropia-Riko.svg)](https://github.com/SakuraEntropia/Entropia-Riko/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Entropia Riko is a professional **node-graph deep-learning editor** — ComfyUI-style
visual workflows for PyTorch (and optional TensorFlow/Keras), with a modular
Blender-like workspace, live training curves, code export, a plugin system, and a
built-in file manager.

It runs as a **web app** (browser) and ships an **Electron shell** so you can use it
as a **standalone desktop app** — you choose.

## Install (PyPI)

```bash
pip install entropia-riko            # core + API server + PyTorch
pip install "entropia-riko[tf]"      # + TensorFlow/Keras nodes
pip install "entropia-riko[hf]"      # + Hugging Face (Diffusers/Transformers) nodes
```

Use it as a Python library:

```python
import entropia_riko
import entropia_riko.nodes          # registers all 194 built-in nodes
from entropia_riko.runtime.registry import default_registry

print(entropia_riko.__version__)                  # "0.1.0"
print(len(default_registry().list()))             # 194
```

Or launch the API server:

```bash
entropia-riko                       # FastAPI on http://127.0.0.1:8000
# equivalent:
python -m uvicorn entropia_riko.server.app:app --port 8000
```

> The pip package ships the **Python runtime** (nodes, executor, codegen,
> trainer, subgraphs, API server). The browser/Electron UI is not in the pip
> package — clone this repo for the full editor.

## Features

- **200+ nodes** — math, tensor ops, neural layers/activations, attention,
  normalization, reductions, shape ops, einsum, losses, data loaders, model
  inference, subgraph references, Hugging Face (Diffusers / Transformers), and
  TensorFlow/Keras equivalents.
- **Node graph canvas** (React Flow) — right-click search menu, drag to connect,
  custom node cards with live output previews.
- **Modular Blender-style workspace** — split/merge/resize any panel (drag the
  corner grip; both resulting rounded windows are previewed in blue), switch any
  window's type, multiple workspace tabs with presets (Layout / Code / Training /
  MNIST Studio / Text→Image / …).
- **Code editor** — a Notepad-style window (File/Edit menus + toolbar: New, Open,
  Save, Undo/Redo, Cut/Copy/Paste) for previewing/editing exported PyTorch code.
- **Train + live loss curve** — stream per-step loss (SSE) into an SVG chart.
- **Clean code export** — PyTorch `nn.Module` and TensorFlow `tf.keras.Model`.
- **Multi-file project export** — File → Export Code → **Export Project…** writes a
  GitHub-layout PyTorch repo (`README.md`, `requirements.txt`, `src/<name>.py`)
  equivalent to the working folder.
- **Subgraph navigation** — double-click a `graph_reference`/`import` node to enter
  it; a Houdini-style breadcrumb (`root / subgraph`) in the top-left shows the
  level and exits back up.
- **Multi-modal subgraph I/O** — `graph_input` / `graph_output` accept a `data_kind`
  (tensor / text / json / image_tensor), not just numbers.
- **Project-as-unit** — work with a project folder (see `templates/project/`), not a
  single file; `.riko`/`.ric` files remain the on-disk format.
- **Asset Library & New File** — a working-directory file manager with
  drag-and-drop folders, right-click create/rename/delete, "expand full nodes"
  (inline a file's graph instead of a subgraph reference), and per-file PyTorch
  code preview.
- **Built-in file explorer** — Windows-style Import/Export (browse, back/forward,
  quick access, recent folders; copy files/folders instead of browser downloads).
- **Plugin system** — load plugins from `.py` files, toggle them on/off; managed in
  both a workspace panel and Preferences.
- **Handwriting pad** — draw a 28×28 digit and send it as a `constant` node to the
  MNIST example for inference.
- **Themes** — Light / Dark / System / **Liquid Glass** (Apple-style translucent).
- **Detachable floating windows** — all dialogs are draggable windows.
- **Binary `.ric`** format + ASCII `.riko` format with full metadata/settings.

## Quick Start (browser)

```bash
cd entropia-riko
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
npm install

# Terminal 1 — API (http://localhost:8000)
.venv/bin/python -m uvicorn entropia_riko.server.app:app --reload --port 8000

# Terminal 2 — frontend (http://localhost:5173)
npm run dev
```

Open **http://localhost:5173** (the `/api` routes proxy to :8000).

## Quick Start (desktop app)

The Electron shell spawns the backend and opens a native window on the Vite dev
server:

```bash
npm install --save-dev electron
npm run dev &                # keep the Vite dev server running
npm run desktop
```

Set `RIKO_DEV_URL` to point at another frontend URL if needed.

## .riko / .ric file format

`.riko` is human-readable JSON; `.ric` is the same document zlib-compressed behind
an `ERIK` magic header. Both carry `version`, `metadata` (name, app, appVersion),
`nodes`, `edges`, and `settings` (theme, background image). See
[`docs/FILE_FORMAT.md`](docs/FILE_FORMAT.md).

## Plugins

Plugins live in `plugins/*/` (a `plugin.json` manifest + an `entry` Python module
that registers nodes via `@register`). Load more from a `.py` file and toggle them
from the **Plugins** panel or **Preferences → Plugins**. Disabled plugins are
skipped so their nodes stay unregistered. Bundled examples: `example_plugin`,
`math_extra`, `stat_extra`.

## Development commands

```bash
.venv/bin/python -m unittest discover -s tests -t .   # Python tests
npm test                                              # frontend tests (vitest)
npm run build                                         # type-check + production build
npm run dev                                           # Vite dev server
npm run desktop                                       # Electron desktop shell
.venv/bin/python scripts/make_brand_assets.py         # brand asset helper (see script)
```

## Distribution / Release

Package a clean, shareable source ZIP (commits pending changes, then archives
only tracked files — no `node_modules`, `.venv`, `dist`, caches, or backups):

```bash
.venv/bin/python scripts/release.py "release note"
```

Output: `entropia-riko-release.zip` in the **parent directory** (the working
folder is never modified). It contains `entropia_riko/`, `public/`, `plugins/`,
`examples/`, `templates/`, `electron/`, `scripts/`, `tests/`, `docs/`, the
READMEs, and config files — everything a recipient needs to `pip install -r
requirements.txt` + `npm install` and run.

### PyPI release

Build and publish the Python package (`entropia-riko` on PyPI):

```bash
.venv/bin/python -m pip install build twine
.venv/bin/python -m build --outdir dist-pypi
.venv/bin/python -m twine upload dist-pypi/*
```

## Project structure

```
entropia_riko/
├── ui/         React app (canvas, panels, code editor, file manager, …)
├── core/       Tensor IR + graph document model (.riko/.ric)
├── runtime/    Registry, executor, PyTorch/TF codegen, trainer, subgraph
├── backend/    Torch device detection + conversion
├── nodes/      Node definitions
├── plugins/    Plugin loader
└── server/     FastAPI API server
plugins/        Bundled plugins
examples/       Ready-to-run pre-wired example graphs (dataset → model → loss → output)
electron/       Desktop shell (main + preload)
scripts/        Brand asset generator
public/brand/   logo.svg + hero.jpg (replace in place to rebrand)
```

## Documentation

- **[User Guide](docs/USER_GUIDE.md)** — full manual (UI, nodes, training, export, API).
- `docs/`: `APP_SPEC.md`, `APP_ARCHITECTURE.md`, `API.md`, `NODE_SYSTEM.md`,
  `DATA_FORMAT.md`, `FILE_FORMAT.md`, `UI_STANDARD.md`, `TORCH_BACKEND.md`,
  `CROSS_PLATFORM.md`.

## License

MIT
