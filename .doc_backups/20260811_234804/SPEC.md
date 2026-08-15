# Product Specification

## Product Name

Torch Node Studio

## Purpose

Torch Node Studio is a standalone node-based application for visual PyTorch workflows.

The app should let users build graphs, connect nodes, configure parameters in a side panel, run the graph, and inspect outputs.

## What This Is

- A standalone UI app.
- A Torch workflow editor.
- A node graph runtime.
- A ComfyUI-style visual execution environment.
- A procedural graph tool for tensor and model workflows.

## What This Is Not

- Not a Houdini plugin.
- Not an HDA generator.
- Not dependent on Houdini APIs.
- Not a wrapper around another DCC application.

## Core Features

- Graph canvas.
- Node library/search.
- Side inspector.
- Execution queue.
- Tensor preview.
- Torch backend.
- Save/load graph documents.
- Cross-platform support.

## First Implementation Scope

Build the minimum useful standalone app prototype:

- App shell.
- Graph canvas.
- Node cards.
- Node selection.
- Side inspector.
- Basic node data model.
- Add and multiply node execution.
- CPU-safe Torch-compatible backend contract.

## Constraints

- Must run on macOS first.
- Must remain compatible with Windows and Linux.
- Must not require GPU.
- Must not require Houdini.
- Must avoid platform-specific paths.
- Must keep UI simple and Material Design-inspired.
