# Design Language

## Purpose

This document describes the intended user experience for the Houdini-facing side of the project.

The system should feel procedural, inspectable, and calm. It should support technical work without hiding important computation behind magic.

## UX Principles

### Procedural First

Every result should be traceable through nodes, inputs, outputs, and parameters.

### Clear Data Flow

Users should be able to understand:

- What enters a node.
- What the node computes.
- What leaves the node.
- Which backend executes the computation.

### Agent-Friendly

The interface and docs should help AI coding agents reason about behavior.

Names should be explicit. Hidden behavior should be avoided.

### Artist-Friendly

The Houdini-facing layer should use familiar concepts:

- Nodes.
- Parameters.
- Inputs and outputs.
- Geometry attributes.
- Visual debugging.

## Node Presentation

Each Houdini-facing node should expose:

- A clear label.
- Short description.
- Input ports.
- Output ports.
- Parameter controls.
- Error state.
- Optional debug view.

## Parameter Design

Parameters should be:

- Named clearly.
- Grouped by purpose.
- Serializable.
- Validated before execution when possible.

Examples:

- Backend settings.
- Shape settings.
- Model settings.
- Debug settings.

## Visual Feedback

The system should eventually show:

- Tensor shape.
- Dtype.
- Device.
- Execution status.
- Error messages.
- Preview summaries.

## Naming Style

Use direct names.

Prefer:

- `Add`
- `Multiply`
- `Linear`
- `Tensor Shape`
- `Device`

Avoid vague names:

- `Processor`
- `Magic Node`
- `AI Thing`
- `Operation 1`

## Debug Experience

Debug output should help both humans and agents.

A good debug report includes:

- Node id.
- Node type.
- Input summary.
- Output summary.
- Backend used.
- Error message if failed.

## Future UI Direction

Future Houdini UI should support:

- Tensor previews.
- Geometry-to-tensor inspection.
- Model input and output summaries.
- Batch inference status.
- Scene-aware world-model debugging.
