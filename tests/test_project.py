"""Project system: manifest, templates, workspace operations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from entropia_riko.project import (
    MANIFEST_FILENAME,
    PROJECT_TEMPLATES,
    REQUIREMENTS_FILENAME,
    create_project,
    is_project,
    migrate_project,
    open_project,
    scan_project,
    validate_project,
)


class TestProjectSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def _create(self, template: str = "computer_vision") -> Path:
        root = self.tmp / "MyProject"
        create_project(root, name="MyProject", template_id=template)
        return root

    def test_templates_available(self) -> None:
        ids = {t.id for t in PROJECT_TEMPLATES}
        self.assertTrue({"empty", "computer_vision", "diffusion", "audio", "video"} <= ids)

    def test_create_generates_manifest(self) -> None:
        root = self._create()
        self.assertTrue(is_project(root))
        self.assertTrue((root / MANIFEST_FILENAME).is_file())
        self.assertTrue((root / REQUIREMENTS_FILENAME).is_file())
        m = open_project(root)
        self.assertEqual(m.name, "MyProject")
        self.assertEqual(m.template, "computer_vision")

    def test_create_generates_structure(self) -> None:
        root = self._create("computer_vision")
        for d in ("datasets/raw", "models/checkpoints", "workflows", "experiments", "configs"):
            self.assertTrue((root / d).is_dir(), d)
        # workflow files seeded
        self.assertTrue((root / "workflows/01_train.riko").is_file())
        self.assertTrue((root / "workflows/02_inference.riko").is_file())

    def test_scan_project(self) -> None:
        root = self._create("audio")
        data = scan_project(root)
        self.assertEqual(data["manifest"]["project"]["name"], "MyProject")
        self.assertTrue(any("00_audio_prepare" in w for w in data["workflows"]))

    def test_validate_ok_and_missing_manifest(self) -> None:
        root = self._create()
        self.assertEqual(validate_project(root), [])
        # a bare folder is not a project
        bare = self.tmp / "bare"
        bare.mkdir()
        self.assertTrue(any("不是" in e or "缺少" in e for e in validate_project(bare)))

    def test_migrate_bare_folder(self) -> None:
        bare = self.tmp / "bare"
        bare.mkdir()
        result = migrate_project(bare)
        self.assertEqual(result["status"], "migrated")
        self.assertTrue((bare / MANIFEST_FILENAME).is_file())

    def test_manifest_roundtrip(self) -> None:
        root = self._create()
        m = open_project(root)
        m.version = "0.2.0"
        m.save(root)
        m2 = open_project(root)
        self.assertEqual(m2.version, "0.2.0")
        self.assertEqual(m2.engine_version, "0.2.0")


if __name__ == "__main__":
    unittest.main()
