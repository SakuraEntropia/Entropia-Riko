"""Workflow management: metadata, categories, and dependency graph.

A project holds multiple workflow documents. This module gives each workflow a
portable metadata block (name, version, I/O types, dependencies) and orders
workflows by their dependency graph so the IDE understands the pipeline.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

from ..core.document import GraphDocument

WORKFLOW_CATEGORIES = (
    "data", "training", "inference", "evaluation", "export", "utility",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WorkflowMetadata:
    """Portable metadata attached to a workflow document."""

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1",
        category: str = "utility",
        dependencies: Sequence[str] = (),
        input_types: Sequence[str] = (),
        output_types: Sequence[str] = (),
        required_models: Sequence[str] = (),
        required_plugins: Sequence[str] = (),
        created: str = "",
        modified: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.version = version
        self.category = category if category in WORKFLOW_CATEGORIES else "utility"
        self.dependencies = list(dependencies)
        self.input_types = list(input_types)
        self.output_types = list(output_types)
        self.required_models = list(required_models)
        self.required_plugins = list(required_plugins)
        self.created = created or _now()
        self.modified = modified or self.created

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "dependencies": self.dependencies,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "required_models": self.required_models,
            "required_plugins": self.required_plugins,
            "created": self.created,
            "modified": self.modified,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> WorkflowMetadata:
        return cls(
            name=str(d.get("name", "workflow")),
            description=str(d.get("description", "")),
            version=str(d.get("version", "1")),
            category=str(d.get("category", "utility")),
            dependencies=d.get("dependencies", []),
            input_types=d.get("input_types", []),
            output_types=d.get("output_types", []),
            required_models=d.get("required_models", []),
            required_plugins=d.get("required_plugins", []),
            created=str(d.get("created", "")),
            modified=str(d.get("modified", "")),
        )


def extract_metadata(doc: GraphDocument) -> WorkflowMetadata:
    """Extract workflow metadata from a graph document's ``metadata`` block."""
    m = doc.metadata or {}
    return WorkflowMetadata.from_dict(m.get("workflow", {}) or {})


def attach_metadata(doc: GraphDocument, meta: WorkflowMetadata) -> None:
    """Attach workflow metadata onto a graph document's ``metadata`` block."""
    doc.metadata = dict(doc.metadata or {})
    doc.metadata["workflow"] = meta.to_dict()


def dependency_order(workflows: Dict[str, WorkflowMetadata]) -> List[str]:
    """Topologically order workflow names by their dependency edges.

    Returns every workflow name; unknown dependencies are ignored (tolerated).
    """
    indegree = {name: 0 for name in workflows}
    dependents: Dict[str, List[str]] = {name: [] for name in workflows}
    for name, meta in workflows.items():
        for dep in meta.dependencies:
            if dep in workflows and dep != name:
                indegree[name] += 1
                dependents[dep].append(name)

    queue = deque(name for name, deg in indegree.items() if deg == 0)
    order: List[str] = []
    while queue:
        name = queue.popleft()
        order.append(name)
        for nxt in dependents[name]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    # Append any remaining (cycles / unresolved) to keep the result complete.
    seen = set(order)
    for name in workflows:
        if name not in seen:
            order.append(name)
    return order
