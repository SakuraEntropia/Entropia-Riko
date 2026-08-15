# Node System

## Purpose

The node system defines how users build procedural deep-learning workflows.
Nodes are visual units of computation with named inputs, outputs, parameters,
and an `execute` method. The runtime wires them together into a dataflow graph,
validates it, orders it topologically, and runs it.

## Node Contract

Defined in `src/nodes/base.py`. Every node subclasses `BaseNode` and declares:

| Class attribute | Meaning |
| --- | --- |
| `type_name` | Stable, unique string identifier (`"add"`, `"linear"`, …). |
| `label` | Human-readable display label. |
| `category` | Grouping for the UI library (`"Math"`, `"Tensor"`, `"Neural"`, …). |
| `inputs` | `list[NodeInput]` — input ports. |
| `outputs` | `list[NodeOutput]` — output ports. |
| `parameters` | `list[Parameter]` — user-editable, serializable parameters. |

And implements:

```python
def execute(self, inputs: dict, params: dict, context: dict) -> dict:
    """Run the node; return {output_port_name: value}."""
```

`inputs` maps input-port names to upstream values; `params` maps parameter names
to their merged values (defaults overlaid with the node's stored parameters);
`context` is a shared dict for graph-wide state (e.g. subgraph `graph_inputs` /
`graph_outputs`). `execute` must return a dict keyed by output-port name.

### Port and parameter classes

- `NodeInput(name, data_kind="tensor", required=True, default=None)` — an input
  port. `required` inputs must be connected (or defaulted); `default` supplies a
  fallback value.
- `NodeOutput(name, data_kind="tensor")` — an output port.
- `Parameter(name, kind="scalar", default=None, required=False, choices=None, dtype=None)`
  — a serializable, user-controlled setting. The constructor merges parameters
  and rejects unknown or missing required ones.

## Registry

Defined in `src/runtime/registry.py`. Maps node type names to node classes.

```python
from runtime.registry import register, default_registry

@register("add")
class AddNode(BaseNode): ...
```

- `register(type_name, node_cls)` — registers a class; raises on empty name or duplicate.
- `get(type_name)` — returns the class; raises `KeyError` for unknown types.
- `list()` — returns sorted type names.
- `unregister(type_name)` — removes a type (no-op if absent); used by plugin toggling.
- `default_registry()` — the process-level singleton used by the executor and server.
- `@register(type_name[, registry])` — class decorator; optional explicit registry.

Built-in nodes are registered by importing `src.nodes`; plugin nodes register
themselves when their entry module is imported.

## Executor

Defined in `src/runtime/executor.py`. Operates on a `GraphDocument` + `Registry`
(no torch, no Houdini).

Pipeline:

1. **`validate(doc, registry) -> list[str]`** — returns error strings (empty = valid):
   - every node `type_name` is registered;
   - every edge references existing nodes and existing ports on the registered classes;
   - every required input is connected;
   - the graph is acyclic (checked via `execution_order`).
2. **`execution_order(doc) -> list[str]`** — Kahn topological sort. Raises
   `RuntimeExecutionError` listing the offending nodes if a cycle exists.
3. **`execute(doc, registry, context) -> {node_id: {port: value}}`** — validates,
   orders, then runs each node in order. Inputs are resolved from upstream
   outputs (or input defaults); each node is instantiated with its parameters,
   `validate_inputs` runs, then `execute` is called. Errors are wrapped in
   `RuntimeExecutionError` with node/port context, and the per-node outputs are
   cached in the returned dict.

## Subgraph Nodes

Defined in `src/runtime/subgraph.py` (node classes live under `src/nodes/subgraph`).

- **`graph_input`** — declares an input to the (sub)graph; becomes a `forward`
  parameter on export.
- **`graph_output`** — declares an output; becomes the `return` value on export.
- **`graph_reference`** — references another `.riko` graph by explicit path
  (`file` parameter).
- **`import`** — imports a graph by module name (`module` parameter), resolved
  like Python modules.

`graph_input` and `graph_output` now carry a `data_kind` parameter
(`tensor | text | json | image_tensor`), making subgraphs **multi-modal** —
they can pass non-tensor payloads (text, JSON, images) across the boundary.

Module resolution (`resolve_graph_file`) searches, in order: absolute paths;
relative paths (against a base dir, then the project root); and bare module
names under `workflows/`, `examples/`, and `examples/models/`. `run_subgraph`
executes the referenced graph with a single input (`graph_input` name=`input`)
and returns `{"output": …}` from its `graph_output` name=`output`.

## Data Kinds

Ports and `TensorValue`s are typed by `data_kind` so the UI and runtime can
distinguish numeric tensors from text/JSON/image/model payloads.

| Kind | Meaning |
| --- | --- |
| `scalar` | A single number (`shape == ()`). |
| `tensor` | A numeric tensor with an inferred shape. |
| `image_tensor` | An image tensor (H×W×C); serialized as base64 PNG previews. |
| `text` | A string payload; **no numeric shape**. |
| `json` | A parsed JSON payload; **no numeric shape**. |
| `model` | A serialized model payload. |

See `DATA_FORMAT.md` for the `TensorValue` field layout and how kinds are
inferred and serialized.
