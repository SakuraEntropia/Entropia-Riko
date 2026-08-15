# Torch Node Studio

Torch Node Studio is a standalone node-based UI application for building and running PyTorch workflows.

It is not a Houdini plugin.

The product uses procedural node-graph thinking and a ComfyUI-style workflow: graph canvas, node library, side inspector, execution queue, and output previews.

## Vision

Create a clean visual workspace where users can compose Torch tensor operations, model inference, and future training workflows through connected nodes.

The application should provide:

- Standalone graph editor.
- Node library.
- Side inspector panel.
- Torch execution backend.
- Tensor previews.
- Graph save/load.
- Cross-platform support for macOS, Windows, and Linux.

## Design Direction

The UI should be simple, clear, and Material Design-inspired.

The first screen should be the actual tool, not a marketing page.

Recommended layout:

```text
Toolbar
Node Library | Graph Canvas | Side Inspector
Status / Queue / Logs
```

## Documentation First

Before implementation, every coding agent should read:

- `SPEC.md`
- `ARCHITECTURE.md`
- `ROADMAP.md`
- `docs/APP_SPEC.md`
- `docs/APP_ARCHITECTURE.md`
- `docs/COMFYUI_WORKFLOW_REFERENCE.md`
- `docs/UI_STANDARD.md`
- `docs/CROSS_PLATFORM.md`
- `docs/API.md`
- `docs/NODE_SYSTEM.md`
- `docs/TORCH_BACKEND.md`
- `docs/DATA_FORMAT.md`

## Current Stage

The project is being refactored from an old integration-oriented concept into a standalone Torch node UI application.

Current priority:

1. Remove old integration-specific language.
2. Define the standalone app architecture.
3. Build the first graph UI prototype.
4. Connect the graph runtime to basic Torch-compatible nodes.
