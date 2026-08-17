"""Project system: manifest, templates, and workspace operations."""

from .manifest import (
    MANIFEST_FILENAME,
    ProjectManifest,
    is_project,
    validate_manifest,
)
from .templates import (
    PROJECT_TEMPLATES,
    REQUIREMENTS_FILENAME,
    ProjectTemplate,
    generate_project,
    get_template,
)
from .workspace import (
    create_project,
    migrate_project,
    open_project,
    scan_project,
    validate_project,
)

__all__ = [
    "ProjectManifest",
    "ProjectTemplate",
    "PROJECT_TEMPLATES",
    "MANIFEST_FILENAME",
    "REQUIREMENTS_FILENAME",
    "is_project",
    "validate_manifest",
    "get_template",
    "generate_project",
    "create_project",
    "open_project",
    "scan_project",
    "validate_project",
    "migrate_project",
]
