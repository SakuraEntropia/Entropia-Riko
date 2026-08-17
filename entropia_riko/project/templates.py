"""Project templates — folder structure + default files for new AI projects.

A template is a declarative list of directories and seed files (workflows,
config, README, requirements). Creating a project materializes the template
under the chosen location and writes a ``project.riko`` manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .manifest import ProjectManifest

REQUIREMENTS_FILENAME = "requirements.riko"


def _identity_workflow(name: str) -> Dict[str, Any]:
    """A minimal valid workflow (graph_input → graph_output identity)."""
    return {
        "version": "1.0",
        "metadata": {
            "name": name,
            "description": f"Default {name} workflow.",
            "inputs": [{"name": "input", "data_kind": "tensor"}],
            "outputs": [{"name": "output", "data_kind": "tensor"}],
        },
        "nodes": [
            {"id": "gin", "type_name": "graph_input", "label": "Input", "category": "Subgraph",
             "position": [100, 150], "parameters": {"name": "input"}, "inputs": [], "outputs": []},
            {"id": "gout", "type_name": "graph_output", "label": "Output", "category": "Subgraph",
             "position": [400, 150], "parameters": {"name": "output"}, "inputs": [], "outputs": []},
        ],
        "edges": [
            {"id": "e0", "source_node": "gin", "source_port": "value",
             "target_node": "gout", "target_port": "value"},
        ],
        "settings": {},
    }


class ProjectTemplate:
    """A declarative project template (directories + seed files)."""

    def __init__(
        self,
        id: str,
        label: str,
        description: str,
        directories: List[str],
        workflows: Dict[str, str],
        files: Dict[str, str],
    ) -> None:
        self.id = id
        self.label = label
        self.description = description
        self.directories = directories
        self.workflows = workflows  # filename (no path) -> workflow name
        self.files = files  # relative path -> content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "directories": self.directories,
            "workflows": list(self.workflows.keys()),
        }


def _workflow_json(name: str) -> str:
    return json.dumps(_identity_workflow(name), indent=2, ensure_ascii=False)


PROJECT_TEMPLATES: List[ProjectTemplate] = [
    ProjectTemplate(
        id="empty",
        label="Empty AI Project",
        description="For custom workflows — a clean project skeleton.",
        directories=[
            "workflows", "datasets", "models", "checkpoints", "cache",
            "bakes", "outputs", "logs", "configs", "scripts", "assets",
        ],
        workflows={"workflows/main.riko": "main"},
        files={"configs/default.yaml": "# Empty AI project configuration\n"},
    ),
    ProjectTemplate(
        id="computer_vision",
        label="Computer Vision",
        description="Image classification / detection: datasets, train, inference, evaluation.",
        directories=[
            "datasets/raw", "datasets/processed", "datasets/annotations",
            "models/pretrained", "models/checkpoints",
            "workflows", "experiments", "logs", "outputs", "configs",
        ],
        workflows={
            "workflows/00_dataset_prepare.riko": "dataset_prepare",
            "workflows/01_train.riko": "train",
            "workflows/02_inference.riko": "inference",
            "workflows/03_evaluation.riko": "evaluation",
        },
        files={
            "configs/training.yaml": "# Training hyperparameters\nepochs: 10\nbatch_size: 32\nlr: 1e-3\n",
            "configs/model.yaml": "# Model architecture\nmodel: cnn\n",
        },
    ),
    ProjectTemplate(
        id="diffusion",
        label="Diffusion / Generative AI",
        description="Text-to-image generation: datasets, training, generation, batch render.",
        directories=[
            "datasets", "models", "checkpoints", "workflows",
            "samples", "outputs", "configs", "cache",
        ],
        workflows={
            "workflows/00_prepare_dataset.riko": "prepare_dataset",
            "workflows/01_training.riko": "training",
            "workflows/02_generation.riko": "generation",
            "workflows/03_batch_render.riko": "batch_render",
        },
        files={"configs/diffusion.yaml": "# Diffusion config\nnum_steps: 30\nguidance_scale: 7.5\n"},
    ),
    ProjectTemplate(
        id="audio",
        label="Audio AI",
        description="Audio datasets, feature extraction, training, generation.",
        directories=[
            "datasets/audio", "datasets/features", "models", "checkpoints",
            "workflows", "outputs", "logs", "configs",
        ],
        workflows={
            "workflows/00_audio_prepare.riko": "audio_prepare",
            "workflows/01_training.riko": "training",
            "workflows/02_generation.riko": "generation",
        },
        files={"configs/audio.yaml": "# Audio config\nsample_rate: 16000\n"},
    ),
    ProjectTemplate(
        id="video",
        label="Video AI",
        description="Video datasets, frame extraction, processing, export.",
        directories=[
            "datasets/videos", "datasets/frames", "models", "checkpoints",
            "workflows", "renders", "outputs", "cache",
        ],
        workflows={
            "workflows/00_frame_extract.riko": "frame_extract",
            "workflows/01_processing.riko": "processing",
            "workflows/02_export.riko": "export",
        },
        files={"configs/video.yaml": "# Video config\nfps: 30\n"},
    ),
]

TEMPLATES_BY_ID = {t.id: t for t in PROJECT_TEMPLATES}


def get_template(template_id: str) -> ProjectTemplate:
    """Return a template by id; raise if unknown."""
    if template_id not in TEMPLATES_BY_ID:
        raise ValueError(
            f"what: 未知项目模板 '{template_id}'。\n"
            f"where: project.templates.get_template\n"
            f"how_to_fix: 可用模板: {sorted(TEMPLATES_BY_ID)}。"
        )
    return TEMPLATES_BY_ID[template_id]


def _requirements_json(manifest: ProjectManifest) -> str:
    return json.dumps(
        {
            "python": manifest.dependencies.get("python", []),
            "models": manifest.dependencies.get("models", []),
            "plugins": manifest.dependencies.get("plugins", []),
            "runtime": {"gpu": manifest.gpu, "cuda": manifest.cuda},
        },
        indent=2,
        ensure_ascii=False,
    )


def generate_project(root: Path, manifest: ProjectManifest) -> Path:
    """Materialize a project under `root`: directories, seed files, manifest."""
    template = get_template(manifest.template)
    root.mkdir(parents=True, exist_ok=True)

    for d in template.directories:
        (root / d).mkdir(parents=True, exist_ok=True)

    for rel, wf_name in template.workflows.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_workflow_json(wf_name) + "\n", encoding="utf-8")

    for rel, content in template.files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    (root / REQUIREMENTS_FILENAME).write_text(_requirements_json(manifest) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        f"# {manifest.name}\n\nGenerated by **Entropia Riko** ({manifest.engine_version}) "
        f"from the '{template.label}' template.\n", encoding="utf-8"
    )
    manifest.workflows.setdefault("all", list(template.workflows.keys()))
    manifest.workflows.setdefault(
        "default", template.workflows[list(template.workflows)[0]] if template.workflows else "workflows/main.riko"
    )
    manifest.save(root)
    return root
