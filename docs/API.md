# API Reference

Entropia Riko is a FastAPI server (`src/server/app.py`). The React UI talks to it
over HTTP; `/api` routes are proxied from the Vite dev server (5173) to the API
server (8000).

## Graph execution & export

| Method | Path | Body / Query | Description |
|--------|------|--------------|-------------|
| GET | `/api/health` | — | Health check → `{"status":"ok"}` |
| GET | `/api/nodes` | — | All registered node definitions (built-in + enabled plugins) |
| POST | `/api/execute` | GraphDocument JSON | Validate + execute; returns `{status, outputs, errors}` |
| POST | `/api/export_python` | GraphDocument JSON | Generate a `torch.nn.Module` script |
| POST | `/api/export_keras` | GraphDocument JSON | Generate a `tf.keras.Model` script |
| POST | `/api/export_binary` | GraphDocument JSON | Encode to `.ric` (base64) for download |
| POST | `/api/export_project` | `{doc, dir}` | Write a multi-file PyTorch repo (`README.md`, `requirements.txt`, `src/<name>.py`) into `dir` |

## Training

| Method | Path | Body / Query | Description |
|--------|------|--------------|-------------|
| POST | `/api/train` | `{doc, steps, lr, wd}` | Train; return loss history once |
| POST | `/api/train/stream` | `{doc, steps, lr, wd}` | Stream per-step loss as NDJSON (live curve) |

## Files (project-root .riko/.ric)

| Method | Path | Body / Query | Description |
|--------|------|--------------|-------------|
| GET | `/api/files` | — | List `.riko`/`.ric` under `workflows/` + `examples/` with import deps |
| GET | `/api/files/content?path=` | — | Read a `.riko`/`.ric` document |
| POST | `/api/files/save` | `{name, format, doc}` | Save to `workflows/<name>.riko|.ric` |
| POST | `/api/files/decode` | raw `.ric` bytes | Decode a binary body into a document |

## Working folder (project-as-unit)

The working folder is the project root for the mini file manager. Defaults to
`workflows/`; importable via `set_root`. A `.riko` cache folder is created inside
it for the tool's own state.

| Method | Path | Body / Query | Description |
|--------|------|--------------|-------------|
| GET | `/api/project/tree` | — | Recursive tree of the working folder (hides dotfiles) |
| POST | `/api/project/set_root` | `{path}` | Import a working folder; persist; create `.riko` cache |
| POST | `/api/project/create` | `{name, dir?}` | Create an empty `.riko` |
| POST | `/api/project/mkdir` | `{name, dir?}` | Create a folder |
| POST | `/api/project/rename` | `{path, newName}` | Rename a file/folder |
| POST | `/api/project/move` | `{path, targetDir}` | Move (drag & drop in the tree) |
| POST | `/api/project/delete` | `{path}` | Delete a file/empty folder |
| POST | `/api/project/code` | `{path}` | Export PyTorch code for a `.riko`/`.ric` |
| GET | `/api/project/open?path=` | — | Read a working-folder document |

## File explorer (import/export via copy)

| Method | Path | Body / Query | Description |
|--------|------|--------------|-------------|
| GET | `/api/fs/list?path=` | — | List a directory (Windows-explorer style) |
| POST | `/api/fs/import` | `{src}` | Copy a file/folder into the working folder |
| POST | `/api/fs/export` | `{src, dest}` | Copy a working-folder file/folder out |
| POST | `/api/fs/save` | `{path, doc, format}` | Save the graph document to an absolute path |

## Plugins

| Method | Path | Body / Query | Description |
|--------|------|--------------|-------------|
| GET | `/api/plugins` | — | List plugins (`enabled`, `status`, registered `nodes`) |
| POST | `/api/plugins/toggle` | `{name, enabled}` | Enable/disable a plugin (unregisters its nodes) |
| POST | `/api/plugins/upload` | `{name, code}` | Install a plugin from Python source |

## Core contracts

- **Graph document**: `version`, `metadata` (name/app/appVersion), `nodes`,
  `edges`, `settings`. See `DATA_FORMAT.md` / `FILE_FORMAT.md`.
- **Node contract**: `type_name`, `label`, `category`, `inputs`, `outputs`,
  `parameters`, `execute(inputs, params, context)`.
- **Runtime**: validate → Kahn topological sort → per-node `execute`.
- **Tensor IR**: `TensorValue` (`data` / `shape` / `dtype` / `device` / `kind` /
  `metadata`), with `scalar|tensor|image_tensor|text|json|model` kinds.
