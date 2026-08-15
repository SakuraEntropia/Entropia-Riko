# Project Template

This is the default **Entropia Riko project** structure — the unit of work is a
*project* (a folder), not a single file.

```
.
├── README.md          # this file
└── src/
    ├── __init__.py
    └── example.riko   # a starter graph (graph_input → graph_output)
```

- Author your graphs as `.riko` (JSON) / `.ric` (binary) files under `src/`.
- Use **File → Export Project…** to emit an equivalent multi-file PyTorch repo
  (`README.md`, `requirements.txt`, `src/<name>.py`) in GitHub layout.
- `.riko` cache folders store the tool's own state.
