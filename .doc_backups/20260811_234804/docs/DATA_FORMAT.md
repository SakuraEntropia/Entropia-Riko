# Data Format

## Purpose

This document defines the data that moves through the graph.

The system needs a small tensor intermediate representation so that Houdini, PyTorch, and future scene formats can communicate without taking ownership of each other.

## Tensor Value

A tensor value should include:

- `data`: the underlying value or array-like payload.
- `shape`: tensor dimensions.
- `dtype`: data type.
- `device`: preferred or actual execution device.
- `metadata`: optional descriptive information.

## Data Kinds

Initial data kinds:

- `tensor`
- `scalar`
- `string`
- `model`
- `geometry`
- `unknown`

The first implementation should focus on `tensor` and `scalar`.

## Metadata

Metadata may include:

- Source node.
- Semantic name.
- Geometry attribute name.
- Coordinate space.
- Batch dimension meaning.
- Scene path.

Metadata should be optional. Core math nodes should not require complex metadata.

## Shape Rules

Nodes should describe shape behavior when possible.

Examples:

- Add: output shape follows broadcasting rules.
- Multiply: output shape follows broadcasting rules.
- Linear: output shape replaces the final input dimension with output features.

## Serialization

Graph data should eventually be serializable.

Preferred properties:

- Plain Python-compatible structures.
- JSON or YAML-friendly metadata.
- No hidden runtime-only state in saved graph definitions.

## Houdini Mapping

Houdini geometry can later map into Tensor IR.

Possible mappings:

- Point positions to tensor.
- Attributes to tensor channels.
- Primitive data to structured tensors.
- Node parameters to scalar inputs.

## USD and USP Mapping

Future scene integration may map:

- Scene paths to metadata.
- Geometry attributes to tensor fields.
- Materials to neural shader nodes.
- World-model outputs to procedural scene updates.

## First Implementation Scope

The first data format should be intentionally small:

- Tensor shape.
- Dtype.
- Device.
- Python or Torch-compatible payload.
- Optional metadata dictionary.
