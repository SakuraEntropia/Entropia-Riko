# Data Format

## Tensor IR — `TensorValue`

Defined in `src/core/tensor.py`. The portable value that flows through the graph
and across the API boundary. Pure Python — it never imports torch; `src/backend`
converts between `TensorValue` and `torch.Tensor`.

### Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `data` | any | Payload: number, nested list, string, parsed JSON, model, … |
| `shape` | `tuple[int, …]` | Numeric shape; `()` for scalars and non-numeric kinds. |
| `dtype` | `str` | E.g. `"float32"` (default). |
| `device` | `str` | E.g. `"cpu"` (default). |
| `metadata` | `dict` | Free-form metadata (e.g. image preview). |
| `kind` / `data_kind` | `str` | One of the data kinds below. |

Constructor: `TensorValue(data, shape=None, dtype="float32", device="cpu", metadata=None, kind=None)`.
When `shape` is omitted it is inferred from the data (numbers → `()`, nested
lists → their nested shape).

### Data kinds

`scalar | tensor | image_tensor | text | json | model`

- `scalar` — `shape == ()`.
- `tensor` — numeric tensor (default when no explicit kind is set).
- `image_tensor` — image tensor; the server serializes it as a base64 PNG data
  URL from `metadata.preview.image` rather than shipping pixel lists.
- `text` — string payload; **no numeric shape** (shape stays `()`).
- `json` — parsed JSON payload; **no numeric shape** (shape stays `()`).
- `model` — serialized model payload.

`data_kind` is derived as follows: an explicit `kind` wins; otherwise the value
is `"scalar"` when `shape == ()` and `"tensor"` otherwise.

### Helpers

- `summary()` — short human-readable preview string for the UI.
- `to_list()` — deep-copy of the payload.
- `item()` — unwrap a scalar (raises if not scalar).
- `infer_shape`, `broadcast_shapes`, `broadcast_op` — shape inference and
  element-wise broadcasting over nested-list payloads.

## Graph Document — `GraphDocument`

Defined in `src/core/document.py`. The saveable workflow: UI graph state
(nodes with positions, edges, ports, settings) serialized to dict/JSON.

### Top-level fields

| Field | Type | Meaning |
| --- | --- | --- |
| `version` | `str` | `"1.0"`. |
| `metadata` | `dict` | `{name, app, appVersion, …}`. |
| `nodes` | list | Serialized nodes (see below). |
| `edges` | list | Serialized edges (see below). |
| `settings` | `dict` | Graph-level settings. |

In memory, `nodes` is a dict keyed by node id; `to_dict()` serializes it as a
list of node objects.

## Serialization

### JSON (`.riko`)

`to_json(indent=2)` produces human-readable ASCII JSON:

```json
{
  "version": "1.0",
  "metadata": { "name": "My Graph", "app": "Entropia Riko", "appVersion": "0.1.0" },
  "nodes": [
    {
      "id": "n1",
      "type_name": "constant",
      "label": "Constant",
      "category": "Inputs",
      "position": [0.0, 0.0],
      "parameters": { "value": 3.0 },
      "inputs": [],
      "outputs": [
        { "name": "value", "label": "value", "data_kind": "scalar", "direction": "out" }
      ]
    }
  ],
  "edges": [
    { "id": "e1", "source_node": "n1", "source_port": "value", "target_node": "n2", "target_port": "left" }
  ],
  "settings": {}
}
```

### Binary (`.ric`)

`to_binary()` produces a compact binary blob:

```text
b"ERIK"  +  b"\x01"  +  zlib.compress(json.dumps(doc, separators=(",", ":")))
```

- `b"ERIK"` — magic header.
- `b"\x01"` — version byte.
- payload — the same dict as JSON, but minified and zlib-compressed.

`from_binary()` reverses this and rejects input missing the `ERIK` magic.

## Node serialization (`NodeModel`)

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | `str` | Unique node id within the document. |
| `type_name` | `str` | Registered node type. |
| `label` | `str` | Display label (defaults to `""`). |
| `category` | `str` | Category (defaults to `""`). |
| `position` | `[x, y]` | Canvas position (serialized as a list; tuple in memory). |
| `parameters` | `dict` | Parameter name → value. |
| `inputs` | list | Input `PortModel`s. |
| `outputs` | list | Output `PortModel`s. |

## Port serialization (`PortModel`)

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | Port name. |
| `label` | `str` | Display label (defaults to `name`). |
| `data_kind` | `str` | `"tensor"` by default; one of the data kinds. |
| `direction` | `str` | `"in"` or `"out"`. |

## Edge serialization (`EdgeModel`)

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | `str` | Unique edge id. |
| `source_node` | `str` | Source node id. |
| `source_port` | `str` | Source output port name. |
| `target_node` | `str` | Target node id. |
| `target_port` | `str` | Target input port name. |
