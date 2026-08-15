"""Entropia Riko API server (FastAPI).

Exposes graph execution and the node registry over HTTP so the React UI
can call the Python runtime. Run:

    uvicorn src.server.app:app --reload --port 8000
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import src.nodes  # noqa: F401  触发节点注册
from ..plugins.loader import (
    load_plugins,
    loaded_plugins,
    set_plugin_enabled,
    upload_plugin,
)
from ..core.document import GraphDocument
from ..core.tensor import TensorValue
from ..runtime.executor import execute, RuntimeExecutionError
from ..runtime.registry import default_registry
from ..runtime.codegen import export_python, export_python_project
from ..runtime.codegen_tf import export_keras
from ..runtime.trainer import train_graph, iter_losses
from ..runtime.subgraph import PROJECT_ROOT, MODULE_SEARCH_PATHS, resolve_graph_file

# Load user plugins (registers any plugin-provided node types).
load_plugins()

app = FastAPI(title="Entropia Riko API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/plugins")
def get_plugins() -> Dict[str, Any]:
    """List loaded user plugins (enabled/disabled status + registered nodes)."""
    return {"plugins": loaded_plugins}


@app.post("/api/plugins/toggle")
def toggle_plugin(body: Dict[str, Any]) -> Dict[str, Any]:
    """Enable or disable a plugin by name (unregisters its nodes when disabled)."""
    name = str(body.get("name", "")).strip()
    enabled = bool(body.get("enabled", True))
    if not name:
        return {"status": "error", "error": "缺少 name"}
    try:
        return {"status": "success", "plugins": set_plugin_enabled(name, enabled)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/plugins/upload")
def upload_plugin_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Install a plugin from raw Python source (the UI reads a .py file client-side)."""
    name = str(body.get("name", "")).strip()
    code = str(body.get("code", ""))
    if not name or not code:
        return {"status": "error", "error": "缺少 name 或 code"}
    try:
        return {"status": "success", "plugins": upload_plugin(name, code)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/nodes")
def get_nodes() -> Dict[str, Any]:
    """Return all registered node definitions for the UI library."""
    reg = default_registry()
    nodes: List[Dict[str, Any]] = []
    for type_name in reg.list():
        cls = reg.get(type_name)
        nodes.append(
            {
                "type": type_name,
                "label": getattr(cls, "label", type_name),
                "category": getattr(cls, "category", ""),
                "inputs": [
                    {"name": i.name, "data_kind": i.data_kind, "required": i.required}
                    for i in cls.inputs
                ],
                "outputs": [
                    {"name": o.name, "data_kind": o.data_kind} for o in cls.outputs
                ],
                "parameters": [
                    {
                        "name": p.name,
                        "kind": p.kind,
                        "default": p.default,
                        "required": p.required,
                        "dtype": p.dtype,
                    }
                    for p in cls.parameters
                ],
            }
        )
    return {"nodes": nodes}


def _serialize_value(tv: TensorValue) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "shape": list(tv.shape),
        "dtype": tv.dtype,
        "device": tv.device,
        "summary": tv.summary(),
        "data_kind": tv.data_kind,
    }
    if tv.data_kind == "image_tensor":
        out["data"] = None  # images are too large to ship as JSON lists
        img = tv.metadata.get("preview", {}).get("image")
        if img:
            out["image"] = img  # base64 PNG data-URL
    else:
        out["data"] = tv.to_list()
    return out


@app.post("/api/execute")
def execute_graph(body: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a graph document (JSON) and return serialized outputs."""
    try:
        doc = GraphDocument.from_dict(body)
        outputs = execute(doc)
        result: Dict[str, Dict[str, Any]] = {}
        for nid, ports in outputs.items():
            result[nid] = {}
            for port, tv in ports.items():
                if isinstance(tv, TensorValue):
                    result[nid][port] = _serialize_value(tv)
                else:
                    result[nid][port] = {"summary": str(tv), "data": tv}
        return {"status": "success", "outputs": result, "errors": []}
    except RuntimeExecutionError as exc:
        return {"status": "error", "outputs": {}, "errors": [str(exc)]}
    except Exception as exc:
        return {"status": "error", "outputs": {}, "errors": [f"{type(exc).__name__}: {exc}"]}


@app.post("/api/export_python")
def export_python_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Generate executable Python code from a graph document."""
    try:
        doc = GraphDocument.from_dict(body)
        code = export_python(doc)
        return {"status": "success", "code": code}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/export_keras")
def export_keras_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a tf.keras.Model script from a graph of TensorFlow/Keras nodes."""
    try:
        doc = GraphDocument.from_dict(body)
        code = export_keras(doc)
        return {"status": "success", "code": code}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/export_binary")
def export_binary_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Encode a graph document to the binary .ric format (base64 for download)."""
    try:
        doc = GraphDocument.from_dict(body)
        return {
            "status": "success",
            "base64": base64.b64encode(doc.to_binary()).decode("ascii"),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/export_project")
def export_project_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Write a multi-file PyTorch project into a directory (GitHub layout)."""
    doc = body.get("doc")
    target_dir = str(body.get("dir", "")).strip()
    if doc is None or not target_dir:
        return {"status": "error", "error": "缺少 doc 或 dir"}
    root = Path(target_dir).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
        files = export_python_project(GraphDocument.from_dict(doc))
        written: List[str] = []
        for f in files:
            p = root / f["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f["content"], encoding="utf-8")
            written.append(f["path"])
        return {"status": "success", "root": str(root), "files": written}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# File manager endpoints (disk-backed .riko files + import relationships).
# ---------------------------------------------------------------------------

_WORKFLOWS_DIR = PROJECT_ROOT / "workflows"


def _safe_project_file(path: Path) -> bool:
    """Only allow .riko / .ric files inside the project root (path guard)."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return (
        resolved.suffix in (".riko", ".ric")
        and str(resolved).startswith(str(PROJECT_ROOT.resolve()) + "/")
        and resolved.is_file()
    )


def _read_doc_dict(p: Path) -> Dict[str, Any]:
    """Read a graph document from either format: .riko (ASCII JSON) or .ric (binary)."""
    if p.suffix == ".ric":
        return GraphDocument.from_binary(p.read_bytes()).to_dict()
    return json.loads(p.read_text(encoding="utf-8"))


def _scan_riko_files() -> List[Dict[str, Any]]:
    """Scan workflows/ and examples/ for .riko/.ric files with their import targets."""
    roots = [PROJECT_ROOT / "workflows", PROJECT_ROOT / "examples"]
    seen: set = set()
    files: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.riko")) + sorted(root.rglob("*.ric")):
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            info: Dict[str, Any] = {
                "name": p.stem,
                "path": rel,
                "format": "binary" if p.suffix == ".ric" else "ascii",
                "imports": [],
            }
            try:
                data = _read_doc_dict(p)
                for n in data.get("nodes", []):
                    if n.get("type_name") in ("graph_reference", "import"):
                        spec = (
                            n.get("parameters", {}).get("file")
                            or n.get("parameters", {}).get("module")
                        )
                        if not spec:
                            continue
                        resolved = resolve_graph_file(spec, base_dir=p.parent)
                        info["imports"].append({
                            "spec": spec,
                            "path": resolved.relative_to(PROJECT_ROOT).as_posix() if resolved else None,
                            "resolved": resolved is not None,
                        })
            except Exception:
                # Malformed file: still list it, just without import metadata.
                pass
            files.append(info)
    return files


@app.get("/api/files")
def list_files() -> Dict[str, Any]:
    """List all .riko files with their import (graph_reference/import) targets."""
    return {"files": _scan_riko_files(), "search_paths": [str(p.relative_to(PROJECT_ROOT)) for p in MODULE_SEARCH_PATHS]}


@app.get("/api/files/content")
def get_file_content(path: str) -> Dict[str, Any]:
    """Return the JSON content of a .riko/.ric file (relative to project root)."""
    target = PROJECT_ROOT / path
    if not _safe_project_file(target):
        return {"status": "error", "error": f"文件不存在或非法: {path}"}
    try:
        doc = _read_doc_dict(target)
        return {"status": "success", "doc": doc}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/files/decode")
async def decode_file(request: Request) -> Dict[str, Any]:
    """Decode an uploaded .ric binary file body into a graph document JSON."""
    try:
        raw = await request.body()
        doc = GraphDocument.from_binary(raw)
        return {"status": "success", "doc": doc.to_dict()}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/train")
def train_graph_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Train a self-contained graph (data loader + loss) and return loss history."""
    try:
        doc = GraphDocument.from_dict(body.get("doc", body))
        steps = int(body.get("steps", 20))
        lr = float(body.get("lr", 1e-3))
        wd = float(body.get("wd", 0.0))
        losses = train_graph(doc, steps=steps, lr=lr)
        return {"status": "success", "losses": losses}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/train/stream")
def train_stream_endpoint(body: Dict[str, Any]):
    """Stream per-step losses (NDJSON) for the live loss curve."""
    doc = GraphDocument.from_dict(body.get("doc", body))
    steps = int(body.get("steps", 20))
    lr = float(body.get("lr", 1e-3))
    wd = float(body.get("wd", 0.0))

    def gen():
        try:
            for item in iter_losses(doc, steps=steps, lr=lr, wd=wd):
                yield json.dumps(item) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            yield json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/files/save")
def save_file(body: Dict[str, Any]) -> Dict[str, Any]:
    """Save a graph document to workflows/<name>.riko (ASCII) or .ric (binary).

    The format is chosen by the name extension (default `.riko`), or by the
    optional `format` field ("ascii" / "binary").
    """
    name = str(body.get("name", "")).strip()
    doc = body.get("doc")
    if not name or doc is None:
        return {"status": "error", "error": "缺少 name 或 doc"}
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_.") or "workflow"

    binary = body.get("format") == "binary" or safe_name.lower().endswith(".ric")
    if not safe_name.lower().endswith((".riko", ".ric")):
        safe_name += ".ric" if binary else ".riko"

    path = _WORKFLOWS_DIR / safe_name
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        gd = GraphDocument.from_dict(doc)
        if binary:
            path.write_bytes(gd.to_binary())
        else:
            path.write_text(gd.to_json(indent=2), encoding="utf-8")
        return {"status": "success", "path": path.relative_to(PROJECT_ROOT).as_posix(), "format": "binary" if binary else "ascii"}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Project (working-directory) mini file manager endpoints.
# ---------------------------------------------------------------------------

# The tool's cache folder inside the project root, and the cache folder (`.riko`)
# that lives inside each imported working folder.
_RIKO_CONFIG_DIR = PROJECT_ROOT / ".riko"
_RIKO_CONFIG_FILE = _RIKO_CONFIG_DIR / "config.json"


def _default_working_root() -> Path:
    return (PROJECT_ROOT / "workflows").resolve()


def _load_working_root() -> Path:
    try:
        if _RIKO_CONFIG_FILE.exists():
            data = json.loads(_RIKO_CONFIG_FILE.read_text(encoding="utf-8"))
            p = Path(str(data.get("working_root", ""))).expanduser().resolve()
            if p.is_absolute() and p.is_dir():
                return p
    except Exception:
        pass
    return _default_working_root()


_working_root: Path = _load_working_root()
_working_root.mkdir(parents=True, exist_ok=True)


def _ensure_cache_dir(root: Path) -> Path:
    """Create the tool's cache folder (`.riko`) inside a working folder."""
    cache = root / ".riko"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _resolve_working(rel: str):
    """Resolve a working-root-relative path (or an absolute path) to a Path."""
    rel = rel.replace("\\", "/").strip("/")
    if not rel:
        return None
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p.resolve()
    base = _working_root.resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return None
    return target


def _walk_tree(root: Path, rel: str = "") -> List[Dict[str, Any]]:
    """Recursively map a directory into {name, path, type, children} entries."""
    items: List[Dict[str, Any]] = []
    if not root.exists():
        return items
    for child in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.startswith("."):
            continue  # hide the .riko cache folder and other dotfiles
        child_rel = f"{rel}/{child.name}" if rel else child.name
        if child.is_dir():
            items.append({
                "name": child.name,
                "path": child_rel,
                "type": "dir",
                "children": _walk_tree(child, child_rel),
            })
        elif child.suffix in (".riko", ".ric"):
            items.append({
                "name": child.name,
                "path": child_rel,
                "type": "file",
            })
    return items


def _read_project_doc(target: Path) -> GraphDocument:
    if target.suffix == ".ric":
        return GraphDocument.from_binary(target.read_bytes())
    return GraphDocument.from_dict(json.loads(target.read_text(encoding="utf-8")))


@app.get("/api/project/tree")
def project_tree() -> Dict[str, Any]:
    """Return the working directory as a recursive tree."""
    return {"root": str(_working_root), "tree": _walk_tree(_working_root)}


@app.post("/api/project/set_root")
def project_set_root(body: Dict[str, Any]) -> Dict[str, Any]:
    """Import a working folder: validate it, create its `.riko` cache, persist."""
    global _working_root
    raw = str(body.get("path", "")).strip()
    if not raw:
        return {"status": "error", "error": "缺少 path"}
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        return {"status": "error", "error": f"不是有效目录: {raw}"}
    _working_root = p
    cache = _ensure_cache_dir(p)
    (cache / "config.json").write_text(
        json.dumps({"app": "Entropia Riko", "working_root": str(p)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _RIKO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _RIKO_CONFIG_FILE.write_text(json.dumps({"working_root": str(p)}, indent=2), encoding="utf-8")
    return {"status": "success", "root": str(p), "tree": _walk_tree(_working_root)}


_STARTER_GRAPH = {
    "version": "1.0",
    "metadata": {
        "name": "example",
        "description": "Starter project graph: graph_input → graph_output (identity).",
        "inputs": [{"name": "input", "data_kind": "tensor"}],
        "outputs": [{"name": "output", "data_kind": "tensor"}],
    },
    "nodes": [
        {"id": "gin", "type_name": "graph_input", "label": "Input", "category": "Subgraph",
         "position": [100, 150], "parameters": {"name": "input", "data_kind": "tensor"},
         "inputs": [], "outputs": []},
        {"id": "gout", "type_name": "graph_output", "label": "Output", "category": "Subgraph",
         "position": [400, 150], "parameters": {"name": "output", "data_kind": "tensor"},
         "inputs": [], "outputs": []},
    ],
    "edges": [{"id": "e0", "source_node": "gin", "source_port": "value",
               "target_node": "gout", "target_port": "value"}],
    "settings": {},
}


def _create_project_tree(root: Path) -> None:
    """Create a PyCharm-style preset project tree under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        f"# {root.name}\n\nGenerated by **Entropia Riko**.\n", encoding="utf-8"
    )
    (root / "requirements.txt").write_text("torch\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.riko/\ndata/\noutputs/\n", encoding="utf-8"
    )
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "__init__.py").write_text('"""Project package."""\n', encoding="utf-8")
    (src / "main.py").write_text(
        '"""Entry point. Export your graph (File → Export Project…) to replace this stub."""\n\n\n'
        'def main():\n    print("Entropia Riko project")\n\n\n'
        'if __name__ == "__main__":\n    main()\n',
        encoding="utf-8",
    )
    graphs = src / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    (graphs / "example.riko").write_text(
        json.dumps(_STARTER_GRAPH, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "data").mkdir(exist_ok=True)
    (root / "outputs").mkdir(exist_ok=True)


@app.post("/api/project/new")
def project_new(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new project folder (PyCharm-style preset tree) and open it."""
    global _working_root
    target = str(body.get("dir", "")).strip()
    if not target:
        return {"status": "error", "error": "缺少 dir"}
    root = Path(target).expanduser().resolve()
    try:
        _create_project_tree(root)
        _working_root = root
        _ensure_cache_dir(root)
        _RIKO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _RIKO_CONFIG_FILE.write_text(
            json.dumps({"working_root": str(root)}, indent=2), encoding="utf-8"
        )
        return {"status": "success", "root": str(root), "tree": _walk_tree(root)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/project/create")
def project_create(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new empty workflow file under the working folder."""
    name = str(body.get("name", "")).strip()
    subdir = str(body.get("dir", "")).strip().replace("\\", "/").strip("/")
    if not name:
        return {"status": "error", "error": "缺少 name"}
    safe = "".join(c for c in name if c.isalnum() or c in "-_.") or "workflow"
    if not safe.lower().endswith((".riko", ".ric")):
        safe += ".riko"
    base = _working_root
    if subdir:
        target_dir = _resolve_working(subdir)
        if target_dir is None or not target_dir.is_dir():
            return {"status": "error", "error": f"目录无效: {subdir}"}
        base = target_dir
    path = base / safe
    if path.exists():
        return {"status": "error", "error": f"文件已存在: {safe}"}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(GraphDocument().to_json(indent=2), encoding="utf-8")
        rel = f"{subdir}/{safe}" if subdir else safe
        return {"status": "success", "path": rel}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/project/mkdir")
def project_mkdir(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a folder under the working folder."""
    name = str(body.get("name", "")).strip()
    subdir = str(body.get("dir", "")).strip().replace("\\", "/").strip("/")
    if not name:
        return {"status": "error", "error": "缺少 name"}
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "folder"
    base = _working_root
    if subdir:
        target_dir = _resolve_working(subdir)
        if target_dir is None or not target_dir.is_dir():
            return {"status": "error", "error": f"目录无效: {subdir}"}
        base = target_dir
    path = base / safe
    try:
        path.mkdir(parents=True, exist_ok=False)
        rel = f"{subdir}/{safe}" if subdir else safe
        return {"status": "success", "path": rel}
    except FileExistsError:
        return {"status": "error", "error": f"已存在: {safe}"}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/project/delete")
def project_delete(body: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a file or empty folder inside the working folder (path-guarded)."""
    rel = str(body.get("path", "")).strip()
    target = _resolve_working(rel)
    if target is None or not target.exists():
        return {"status": "error", "error": f"不存在或非法: {rel}"}
    try:
        if target.is_dir():
            target.rmdir()
        else:
            target.unlink()
        return {"status": "success", "path": rel}
    except OSError as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/project/rename")
def project_rename(body: Dict[str, Any]) -> Dict[str, Any]:
    """Rename a file or folder inside the working folder."""
    rel = str(body.get("path", "")).strip()
    new_name = str(body.get("newName", "")).strip()
    target = _resolve_working(rel)
    if target is None or not target.exists():
        return {"status": "error", "error": f"不存在或非法: {rel}"}
    if not new_name or "/" in new_name or "\\" in new_name:
        return {"status": "error", "error": "非法名称"}
    try:
        target.rename(target.with_name(new_name))
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        new_rel = f"{parent}/{new_name}" if parent else new_name
        return {"status": "success", "path": new_rel}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/project/move")
def project_move(body: Dict[str, Any]) -> Dict[str, Any]:
    """Move a file/folder into another folder (drag & drop in the file tree)."""
    rel = str(body.get("path", "")).strip()
    target_dir = str(body.get("targetDir", "")).strip().replace("\\", "/").strip("/")
    src = _resolve_working(rel)
    if src is None or not src.exists():
        return {"status": "error", "error": f"不存在或非法: {rel}"}
    base = _working_root if not target_dir else _resolve_working(target_dir)
    if base is None or not base.is_dir():
        return {"status": "error", "error": f"目标目录无效: {target_dir}"}
    dest = base / src.name
    if dest.exists():
        return {"status": "error", "error": f"已存在: {src.name}"}
    try:
        src.rename(dest)
        new_rel = f"{target_dir}/{src.name}" if target_dir else src.name
        return {"status": "success", "path": new_rel}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/project/code")
def project_code(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return the exported PyTorch code for a .riko/.ric file in the working folder."""
    rel = str(body.get("path", "")).strip()
    target = _resolve_working(rel)
    if target is None or not target.is_file():
        return {"status": "error", "error": f"不存在或非法: {rel}"}
    try:
        code = export_python(_read_project_doc(target))
        return {"status": "success", "code": code}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/project/open")
def project_open(path: str) -> Dict[str, Any]:
    """Return the JSON content of a .riko/.ric file in the working folder."""
    rel = str(path).strip()
    target = _resolve_working(rel)
    if target is None or not target.is_file():
        return {"status": "error", "error": f"不存在或非法: {rel}"}
    try:
        return {"status": "success", "doc": _read_project_doc(target).to_dict()}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Built-in file explorer (browse + import/export file or folder via copy).
# ---------------------------------------------------------------------------

@app.get("/api/fs/list")
def fs_list(path: str = "") -> Dict[str, Any]:
    """List a directory (Windows-explorer style) for the built-in file explorer."""
    p = Path(path or "~").expanduser().resolve()
    if not p.is_dir():
        return {"status": "error", "error": f"不是目录: {path}"}
    try:
        entries: List[Dict[str, Any]] = []
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if child.name.startswith("."):
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "type": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else 0,
            })
        return {"status": "success", "path": str(p), "parent": str(p.parent), "entries": entries}
    except PermissionError as exc:
        return {"status": "error", "error": f"PermissionError: {exc}"}


@app.post("/api/fs/import")
def fs_import(body: Dict[str, Any]) -> Dict[str, Any]:
    """Copy an external file/folder into the working folder (import)."""
    src = str(body.get("src", "")).strip()
    if not src:
        return {"status": "error", "error": "缺少 src"}
    s = Path(src).expanduser().resolve()
    if not s.exists():
        return {"status": "error", "error": f"不存在: {src}"}
    dest = _working_root / s.name
    if dest.exists():
        return {"status": "error", "error": f"已存在: {dest.name}"}
    try:
        if s.is_dir():
            shutil.copytree(s, dest)
        else:
            shutil.copy2(s, dest)
        return {"status": "success", "path": str(dest)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/fs/export")
def fs_export(body: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a working-folder file/folder to a chosen destination directory (export)."""
    src = str(body.get("src", "")).strip()
    dest_dir = str(body.get("dest", "")).strip()
    if not src or not dest_dir:
        return {"status": "error", "error": "缺少 src 或 dest"}
    s = _resolve_working(src)
    if s is None or not s.exists():
        return {"status": "error", "error": f"工作目录中不存在: {src}"}
    d = Path(dest_dir).expanduser().resolve()
    if not d.is_dir():
        return {"status": "error", "error": f"目标目录无效: {dest_dir}"}
    dest = d / s.name
    try:
        if s.is_dir():
            shutil.copytree(s, dest)
        else:
            shutil.copy2(s, dest)
        return {"status": "success", "path": str(dest)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/fs/save")
def fs_save(body: Dict[str, Any]) -> Dict[str, Any]:
    """Save the graph document to an absolute path (file-manager style save)."""
    path = str(body.get("path", "")).strip()
    doc = body.get("doc")
    if not path or doc is None:
        return {"status": "error", "error": "缺少 path 或 doc"}
    binary = body.get("format") == "binary"
    p = Path(path).expanduser().resolve()
    if binary:
        if p.suffix.lower() != ".ric":
            p = p.with_suffix(".ric")
    elif p.suffix.lower() != ".riko":
        p = p.with_suffix(".riko")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        gd = GraphDocument.from_dict(doc)
        if binary or p.suffix.lower() == ".ric":
            p.write_bytes(gd.to_binary())
        else:
            p.write_text(gd.to_json(indent=2), encoding="utf-8")
        return {"status": "success", "path": str(p), "name": p.stem}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
