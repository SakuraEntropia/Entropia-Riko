"""Entropia Riko API server (FastAPI).

Assembles the API routers over the runtime. Run:

    uvicorn entropia_riko.server.app:app --reload --port 8000

Route modules live in `entropia_riko/server/routers/`, one per feature area;
shared working-directory state lives in `entropia_riko/server/state.py`.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import entropia_riko  # noqa: F401  (__version__)
import entropia_riko.nodes  # noqa: F401  触发全部内置节点注册

from ..plugins.loader import load_plugins
from .routers import files, fs, graph, health, plugins, project, train

# Load user plugins (registers any plugin-provided node types).
load_plugins()

app = FastAPI(title="Entropia Riko API", version=entropia_riko.__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(plugins.router)
app.include_router(graph.router)
app.include_router(files.router)
app.include_router(train.router)
app.include_router(project.router)
app.include_router(fs.router)


def main() -> None:
    """Entry point for the ``entropia-riko`` console script."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
