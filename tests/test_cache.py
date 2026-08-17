"""Bake/cache system: artifact provenance + cache reuse."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from entropia_riko.project import (
    bake_artifact,
    get_bake_artifact,
    get_bake_metadata,
    is_cache_valid,
    list_bakes,
)


class TestCache(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def test_bake_writes_artifact_and_metadata(self) -> None:
        p = bake_artifact(
            self.tmp, "render_001", b"PNGDATA",
            source_node="save_image", parameters={"seed": 1},
        )
        self.assertTrue(p.is_file())
        meta = get_bake_metadata(self.tmp, "render_001")
        self.assertEqual(meta["source_node"], "save_image")
        self.assertEqual(meta["parameters"], {"seed": 1})

    def test_cache_hit_and_miss(self) -> None:
        bake_artifact(self.tmp, "ckpt", b"x", "checkpoint_save", {"lr": 1e-3})
        self.assertTrue(is_cache_valid(self.tmp, "ckpt", {"lr": 1e-3}))
        self.assertFalse(is_cache_valid(self.tmp, "ckpt", {"lr": 1e-4}))
        self.assertFalse(is_cache_valid(self.tmp, "missing", {}))

    def test_list_bakes(self) -> None:
        bake_artifact(self.tmp, "a", b"1", "n", {})
        bake_artifact(self.tmp, "b", b"2", "n", {})
        names = {b["name"] for b in list_bakes(self.tmp)}
        self.assertEqual(names, {"a", "b"})

    def test_get_artifact(self) -> None:
        bake_artifact(self.tmp, "a", b"hello", "n", {})
        art = get_bake_artifact(self.tmp, "a")
        self.assertEqual(art.read_bytes(), b"hello")
        self.assertIsNone(get_bake_artifact(self.tmp, "nope"))


if __name__ == "__main__":
    unittest.main()
