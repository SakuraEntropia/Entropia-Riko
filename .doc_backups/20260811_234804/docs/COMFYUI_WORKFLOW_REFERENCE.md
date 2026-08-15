# ComfyUI-Style Workflow Reference

## Purpose

This document defines which workflow ideas Torch Node Studio should borrow from ComfyUI-style systems.

The goal is not to copy ComfyUI. The goal is to learn from its graph-first workflow and apply that pattern to Torch computation.

## Useful Ideas To Borrow

### Graph As Main Workspace

The graph should be the primary interface.

Users should immediately see:

- Nodes.
- Edges.
- Execution controls.
- Preview area.
- Inspector or settings panel.

### Node Search

Users should add nodes from a searchable node menu.

Expected behavior:

- Search by node name.
- Browse by category.
- Insert node at cursor or canvas center.
- Show short node descriptions.

### Queue-Based Execution

Execution should be explicit and inspectable.

Expected behavior:

- Run graph.
- Queue graph.
- Cancel execution.
- Show running node.
- Show completed nodes.
- Show failed nodes.

### Output Preview

Outputs should be easy to inspect.

Preview types:

- Tensor summary.
- Scalar value.
- Image tensor.
- Text output.
- Model output.
- Debug information.

### Reproducible Workflow

The graph should be saveable and shareable.

Saved workflows should include:

- Nodes.
- Edges.
- Parameters.
- App version.
- Optional runtime settings.

## Differences From ComfyUI

Torch Node Studio is broader than image generation.

It should support:

- Generic tensor operations.
- Neural network components.
- Model inference.
- Training utilities later.
- Data transformation.
- Visual debugging.

Do not assume every workflow produces an image.

## Interaction Model

Basic interaction:

1. Add node.
2. Configure node in side panel.
3. Connect ports.
4. Validate graph.
5. Run graph.
6. Inspect outputs.
7. Save workflow.

## UI Layout

Recommended first layout:

```text
┌────────────────────────────────────────────┐
│ Toolbar                                    │
├──────────────┬─────────────────┬───────────┤
│ Node Library │ Graph Canvas    │ Inspector │
│              │                 │           │
├──────────────┴─────────────────┴───────────┤
│ Status / Queue / Logs                       │
└────────────────────────────────────────────┘
```

## Node Card

A node card should show:

- Title.
- Type/category.
- Input ports.
- Output ports.
- Execution state.
- Error marker.

Detailed parameters belong in the side inspector.

## Side Inspector

The side inspector should show:

- Selected node name.
- Description.
- Parameters.
- Inputs.
- Outputs.
- Preview.
- Debug data.

## Agent Rule

When building UI, the agent should prioritize:

- Usable graph editor first.
- Real controls over placeholder marketing.
- Clear execution feedback.
- Material Design-style simplicity.
- Dense but readable technical panels.
