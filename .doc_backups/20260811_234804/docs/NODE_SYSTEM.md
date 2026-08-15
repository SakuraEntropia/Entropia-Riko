# Node System

## Purpose

The node system defines how computation is represented, connected, validated, and executed.

It should feel inspired by Houdini's procedural graph model while remaining independent from Houdini itself.

## Node Definition

A node is a small unit of computation.

Each node should define:

- A stable type name.
- A human-readable label.
- Named inputs.
- Named outputs.
- Parameters.
- Execution behavior.
- Optional metadata.

## Node Inputs

Inputs represent values required by the node.

Each input should include:

- Name.
- Expected data kind.
- Whether it is required.
- Optional default value.
- Shape constraints when relevant.

## Node Outputs

Outputs represent values produced by the node.

Each output should include:

- Name.
- Data kind.
- Shape rule when known.
- Metadata rule when known.

## Parameters

Parameters are user-controlled settings.

Examples:

- Scalar value.
- Tensor shape.
- Activation type.
- Model path.
- Device preference.

Parameters should be serializable so that graphs can be saved later.

## Execution Model

The graph should execute nodes in dependency order.

Execution should follow this flow:

1. Validate graph connections.
2. Resolve execution order.
3. Gather input values for each node.
4. Execute the node.
5. Store output values.
6. Report errors with node context.

## Node Registry

The registry maps node type names to node classes or factories.

The registry should support:

- Registering a node type.
- Looking up a node by type name.
- Listing available node types.
- Preventing accidental duplicate registrations.

## First Nodes

The first implementation should include simple math nodes:

- `add`
- `multiply`

These nodes are intentionally small. They prove the graph contract before neural or Houdini-specific features are added.

## Design Rules

- Nodes should be composable.
- Nodes should not directly manage graph execution.
- Nodes should not contain Houdini-specific code.
- Nodes should validate their own local inputs.
- Shared validation belongs in the graph layer.

## Error Handling

Errors should include:

- Node id.
- Node type.
- Input or parameter name when relevant.
- Short explanation.
- Suggested fix when obvious.
