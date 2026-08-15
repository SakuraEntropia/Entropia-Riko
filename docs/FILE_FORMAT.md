# Entropia Riko Graph Document Format

A graph document is the unit of work in Entropia Riko: it fully describes a
node graph (its metadata, nodes, edges, and settings). The same logical document
is the object the editor loads, edits, saves, sends to the execution API, and
exports to Python.

The document has **two on-disk representations**, which are interchangeable:

| Extension | Kind            | Description                                                        |
|-----------|-----------------|--------------------------------------------------------------------|
| `.riko`   | ASCII JSON      | Human-readable, diff-friendly, UTF-8 JSON (indented).              |
| `.ric`    | Binary          | `ERIK` magic + version byte + zlib-compressed JSON (compact).      |

Both decode to the *identical* JSON structure described below.

---

## Top-level schema

```json
{
  "version": "1.0",
  "metadata": { "name": "classifier", "app": "entropia-riko", "appVersion": "0.1.0" },
  "nodes": [ ... ],
  "edges": [ ... ],
  "settings": { "theme": "dark", "backgroundImage": "" }
}
```

| Field        | Type    | Required | Description                                              |
|--------------|---------|----------|----------------------------------------------------------|
| `version`    | string  | ✓        | Format version. Currently `"1.0"`.                       |
| `metadata`   | object  |          | Document metadata (name, origin app, app version).       |
| `nodes`      | array   | ✓        | Array of node objects (see [Nodes](#nodes)).             |
| `edges`      | array   | ✓        | Array of edge objects (see [Edges](#edges)).             |
| `settings`   | object  |          | Editor/UI settings (theme, background image).            |

The Python model is `src/core/document.py` (`GraphDocument`). In memory nodes are
keyed by id in a dict; when serialized they are emitted as a plain array. Missing
optional fields are tolerated on load (`version` defaults to `"1.0"`, `metadata`
and `settings` to `{}`).

---

## `metadata` — object, optional

Descriptive information written by the editor:

```json
{
  "name": "classifier",
  "app": "entropia-riko",
  "appVersion": "0.1.0"
}
```

| Field        | Type   | Description                                                    |
|--------------|--------|----------------------------------------------------------------|
| `name`       | string | The document/active-file name (falls back to `"untitled"`).    |
| `app`        | string | The application that produced it: `"entropia-riko"`.           |
| `appVersion` | string | The producing app version (`src/ui/version.ts`).               |

`metadata` is stored as a free-form dict, so consumers must treat unknown keys as
optional (for example, `saveFileToDisk` also writes an empty `description`).
Subgraph I/O is **not** declared here — it is declared by `graph_input` /
`graph_output` nodes in the graph itself (see
[Subgraph references](#subgraph-references)).

---

## `settings` — object, optional

Editor/UI state persisted with the document:

```json
{
  "theme": "dark",
  "backgroundImage": "https://example.com/cover.jpg"
}
```

| Field             | Type   | Description                                                          |
|-------------------|--------|----------------------------------------------------------------------|
| `theme`           | string | UI theme: `light`, `dark`, `system`, or `glass` (Liquid Glass).      |
| `backgroundImage` | string | Optional cover-image URL; empty string means none.                   |

Both are read from `localStorage` (`entropia_riko_theme`,
`entropia_riko_background`) when the editor serializes the document. The backend
does not interpret them; they round-trip unchanged.

---

## Nodes

Each element of `nodes`:

```json
{
  "id": "fc1",
  "type_name": "linear",
  "label": "Linear 8→16",
  "category": "Neural",
  "position": [240, 240],
  "parameters": { "in_features": 8, "out_features": 16 },
  "inputs": [],
  "outputs": []
}
```

| Field        | Type               | Required | Description                                                            |
|--------------|--------------------|----------|------------------------------------------------------------------------|
| `id`         | string             | ✓        | Unique node id within the document.                                    |
| `type_name`  | string             | ✓        | Registered node type (see `GET /api/nodes`).                           |
| `label`      | string             |          | Display name; falls back to the node type's label.                     |
| `category`   | string             |          | UI grouping (falls back to the type's category).                       |
| `position`   | `[number, number]` |          | Canvas position `[x, y]` (defaults to `[0, 0]`).                       |
| `parameters` | object             |          | Parameter values keyed by name; merged over the type's defaults.       |
| `inputs`     | array              |          | Port annotations (see below); usually empty in editor-written files.   |
| `outputs`    | array              |          | Port annotations; usually empty in editor-written files.               |

The editor serializes `inputs`/`outputs` as empty arrays and re-attaches port
definitions from the node registry on load. The model
(`src/core/document.py` `PortModel`) also supports explicit port objects for
hand-authored files:

```json
{ "name": "x", "label": "X", "data_kind": "tensor", "direction": "in" }
```

`data_kind` is one of `scalar`, `tensor`, `image_tensor`, `model`, `text`,
`unknown` (see `src/core/tensor.py` `DATA_KINDS`); `direction` is `"in"` or
`"out"`.

---

## Edges

Each element of `edges`:

```json
{ "id": "e1", "source_node": "a", "source_port": "value",
  "target_node": "b", "target_port": "left" }
```

| Field         | Type   | Required | Description                     |
|---------------|--------|----------|---------------------------------|
| `id`          | string | ✓        | Unique edge id.                 |
| `source_node` | string | ✓        | Source node id.                 |
| `source_port` | string | ✓        | Source node's output port name. |
| `target_node` | string | ✓        | Target node id.                |
| `target_port` | string | ✓        | Target node's input port name.  |

---

## `.riko` — human-readable JSON

- Encoding: UTF-8.
- Serialization: `json.dumps(..., ensure_ascii=False, indent=2)` — readable and
  diff-friendly.
- The editor downloads `.riko` files with MIME type `application/json`.
- Loaded by plain `json.loads(...)` on both the frontend and the backend.

---

## `.ric` — binary

The binary form wraps the *same* JSON document in a compact, compressed
container:

```
offset  size  content
0       4     magic     b"ERIK"
4       1     version   b"\x01"        (container version byte)
5       …     payload   zlib.compress(utf8 JSON)
```

- The JSON payload is serialized compactly (`separators=(",", ":")`,
  `ensure_ascii=False`) before compression, so `.ric` files are smaller than
  their `.riko` counterparts.
- **Magic** `b"ERIK"` identifies the format; **version byte** `b"\x01"` reserves
  space for future container changes (it is *not* the document's `"1.0"`
  `version` string).
- Decoding: verify the leading `ERIK` magic, then
  `zlib.decompress(data[5:])` and `json.loads` the result. A payload without the
  magic header raises `ValueError` ("not a valid .ric binary").

This is implemented as `GraphDocument.to_binary()` / `from_binary()` in
`src/core/document.py`:

```python
_RIC_MAGIC    = b"ERIK"
_RIC_VERSION  = b"\x01"

def to_binary(self) -> bytes:
    payload = json.dumps(self.to_dict(), ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return _RIC_MAGIC + _RIC_VERSION + zlib.compress(payload)
```

---

## Complete example

```json
{
  "version": "1.0",
  "metadata": { "name": "classifier", "app": "entropia-riko", "appVersion": "0.1.0" },
  "nodes": [
    { "id": "gin",  "type_name": "graph_input", "label": "Input",  "position": [40, 240],   "parameters": { "name": "input" } },
    { "id": "fc1",  "type_name": "linear", "label": "Linear 8→16", "position": [240, 240],  "parameters": { "in_features": 8, "out_features": 16 } },
    { "id": "a1",   "type_name": "relu",   "label": "ReLU",        "position": [440, 240] },
    { "id": "fc2",  "type_name": "linear", "label": "Linear 16→4", "position": [640, 240],  "parameters": { "in_features": 16, "out_features": 4 } },
    { "id": "sm",   "type_name": "softmax", "label": "Softmax",    "position": [840, 240] },
    { "id": "gout", "type_name": "graph_output", "label": "Output", "position": [1040, 240], "parameters": { "name": "output" } }
  ],
  "edges": [
    { "id": "e0", "source_node": "gin",  "source_port": "value",  "target_node": "fc1", "target_port": "x" },
    { "id": "e1", "source_node": "fc1",  "source_port": "output", "target_node": "a1",  "target_port": "x" },
    { "id": "e2", "source_node": "a1",   "source_port": "output", "target_node": "fc2", "target_port": "x" },
    { "id": "e3", "source_node": "fc2",  "source_port": "output", "target_node": "sm",  "target_port": "x" },
    { "id": "e4", "source_node": "sm",   "source_port": "result", "target_node": "gout", "target_port": "value" }
  ],
  "settings": { "theme": "system", "backgroundImage": "" }
}
```

---

## Subgraph references

A `.riko` graph can be referenced as a subgraph by another graph:

- `graph_reference` node — references by a relative/absolute **path**
  (parameter `file`).
- `import` node — references by **module name** (parameter `module`), resolved
  against the module search paths `workflows/`, `examples/`, `examples/models/`.

The referenced document declares its interface with `graph_input`
(`name="input"`) and `graph_output` (`name="output"`) nodes, so graphs compose
like function calls / Python imports. Resolution lives in
`src/runtime/subgraph.py` (`resolve_graph_file`).

---

## Versioning & migration

- The current format version is `"1.0"`; loaders accept it and tolerate missing
  optional fields (empty `metadata`/`settings`, missing `position`, etc.).
- Future *minor* additions must remain backward-compatible (new optional keys).
- *Breaking* changes bump the major version and require an explicit migration.
- The `.ric` container has its own one-byte version (`b"\x01"`) independent of
  the document `version`.

## Related APIs

- `POST /api/execute` — execute a document.
- `POST /api/export_python` / `POST /api/export_keras` — export code.
- `POST /api/export_binary` — encode a document to `.ric` (returned as base64).
- `POST /api/files/decode` — decode an uploaded `.ric` body to a document.
- `GET /api/files`, `GET /api/files/content?path=` — list / read disk-backed
  `.riko`/`.ric` files.
- `POST /api/files/save` — save to `workflows/<name>.riko`/`.ric`.
- `POST /api/fs/save` — save to an arbitrary path (file-picker style).
- `GET /api/project/tree`, `GET /api/project/open`, `POST /api/project/create`,
  … — working-folder mini file manager.
