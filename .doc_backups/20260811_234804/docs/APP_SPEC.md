# Product Specification

## Product Name

Torch Node Studio

## Purpose

Torch Node Studio is a standalone node-based application for building, running, and inspecting PyTorch workflows through a visual graph interface.

The product combines:

- A ComfyUI-style graph workflow.
- Houdini-like procedural node logic.
- A clean Material Design-inspired interface.
- A Torch backend for tensors, models, inference, and future training workflows.

It is not a Houdini plugin.

## Product Goal

Let users build Torch workflows visually.

Users should be able to add nodes, connect ports, edit parameters in a side panel, run the graph, inspect tensor outputs, and save reusable workflows.

## Target Users

### AI Builder

Wants a visual way to compose model inference and tensor operations.

### Technical Artist

Wants procedural node logic without needing Houdini as the host application.

### ML Developer

Wants rapid prototyping of Torch computation graphs with inspectable inputs, outputs, and intermediate tensors.

### AI Coding Agent

Needs clear docs and contracts to implement features without architectural drift.

## Core Product Features

### Graph Canvas

The main workspace where users create and connect nodes.

Required capabilities:

- Add nodes.
- Move nodes.
- Select nodes.
- Connect output ports to input ports.
- Delete nodes and edges.
- Pan and zoom.
- Show validation state.
- Show execution state.

### Side Inspector Panel

The right-side panel edits the selected node.

Required sections:

- Node summary.
- Parameters.
- Inputs.
- Outputs.
- Tensor preview.
- Execution status.
- Debug messages.

### Node Library

Users should be able to search and insert available nodes.

Initial categories:

- Inputs.
- Math.
- Tensor.
- Neural.
- Model.
- Utility.

### Execution Queue

Graph execution should be explicit.

The UI should support:

- Run graph.
- Stop execution.
- Queue status.
- Node running state.
- Node success state.
- Node error state.

### Torch Backend

The backend should support:

- CPU execution.
- CUDA when available.
- MPS on supported Apple systems.
- Tensor conversion.
- Model loading later.
- Inference later.

CPU is the guaranteed baseline.

## Non-Goals

The first version should not implement:

- Houdini integration.
- HDA generation.
- USD integration.
- USP integration.
- World-model tools.
- Full training dashboard.
- Cloud execution.

These may be future phases, but they are not part of the current product direction.

## First Implementation Scope

Stage 1 should build a standalone UI prototype plus core runtime skeleton.

Minimum useful version:

- App shell.
- Graph canvas.
- Example node cards.
- Side inspector.
- Node data model.
- Basic graph state.
- Basic Torch-compatible runtime contract.
- Add and multiply node execution.

## Success Criteria

- The first screen is the actual graph editor.
- The app runs locally on macOS.
- Code remains compatible with Windows and Linux.
- No Houdini dependency exists.
- Nodes can be added, selected, connected, and inspected.
- A simple graph can execute through the runtime.
