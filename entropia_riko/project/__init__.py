"""Project system: manifest, templates, workspace, cache, workflow, experiments."""

from .cache import (
    bake_artifact,
    get_bake_artifact,
    get_bake_metadata,
    is_cache_valid,
    list_bakes,
)
from .experiment import (
    hardware_info,
    list_experiments,
    record_experiment,
)
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
from .workflow import (
    WORKFLOW_CATEGORIES,
    WorkflowMetadata,
    attach_metadata,
    dependency_order,
    extract_metadata,
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
    "WorkflowMetadata",
    "PROJECT_TEMPLATES",
    "WORKFLOW_CATEGORIES",
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
    "record_experiment",
    "list_experiments",
    "hardware_info",
    "bake_artifact",
    "get_bake_artifact",
    "get_bake_metadata",
    "is_cache_valid",
    "list_bakes",
    "extract_metadata",
    "attach_metadata",
    "dependency_order",
]
