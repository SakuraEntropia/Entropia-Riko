# API Design

## Purpose

This document defines the first public contracts for the core system.

The API should be small enough for a coding agent to implement cleanly, but stable enough that future nodes and Houdini adapters can build on it.

## Core Objects

### Tensor Value

Represents data flowing through nodes.

Expected fields:

- `data`
- `shape`
- `dtype`
- `device`
- `metadata`

### Node

Base behavior for all nodes.

Expected capabilities:

- Declare inputs.
- Declare outputs.
- Declare parameters.
- Validate local inputs.
- Execute with resolved inputs.

### Graph

Owns nodes and connections.

Expected capabilities:

- Add node.
- Remove node.
- Connect node outputs to node inputs.
- Validate graph.
- Execute graph.

### Registry

Maps node type names to node implementations.

Expected capabilities:

- Register node.
- Get node.
- List nodes.

## Suggested Python Shape

This is design guidance, not final code.

```python
graph = Graph()
graph.add_node("a", type="constant", params={"value": 1.0})
graph.add_node("b", type="constant", params={"value": 2.0})
graph.add_node("sum", type="add")
graph.connect("a", "value", "sum", "left")
graph.connect("b", "value", "sum", "right")
result = graph.execute()
```

## Node Contract

Each node should provide:

- `type_name`
- `inputs`
- `outputs`
- `parameters`
- `execute(inputs, params, context)`

## Parameter Fields

A `Parameter` describes a user-controlled setting. Fields:

- `name`: parameter name.
- `kind`: coarse value category, e.g. `scalar` or `any` (tensor/list payload).
- `default`: default value (may be `None`).
- `required`: whether the user must supply a value.
- `choices`: optional allowed-value list (maps to a menu in adapters).
- `dtype`: optional platform-neutral type hint (`int` / `float` / `string` / `bool` / `data`). Used by adapters (e.g. Houdini mapping) to infer the control type when no `default` is available; omitted values fall back to inference from `default`.

## Execution Context

The execution context may include:

- Backend.
- Device.
- Logger.
- Runtime metadata.
- Debug flags.

## Error Contract

Errors should be readable by humans and useful to agents.

Good errors include:

- What failed.
- Which node failed.
- Which input or parameter caused the issue.
- How to fix it when obvious.

## Compatibility Rule

Avoid changing public names casually after Stage 1.

If an API must change, update:

- `SPEC.md`
- `ARCHITECTURE.md`
- This file
- Relevant tests
