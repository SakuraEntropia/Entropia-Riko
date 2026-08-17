"""Professional I/O nodes: file_input, dataset, checkpoint save/load."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

import entropia_riko.nodes  # noqa: F401  触发节点注册
from entropia_riko.nodes.torch_ops.io_nodes import (
    CheckpointLoadNode,
    CheckpointSaveNode,
    DatasetNode,
    FileInputNode,
)
from entropia_riko.runtime.registry import default_registry


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


class TestIONodes(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.reg = default_registry()

    def test_registered(self) -> None:
        for t in ("file_input", "dataset", "checkpoint_save", "checkpoint_load"):
            self.assertIn(t, self.reg.list())

    def test_file_input_file(self) -> None:
        f = self.tmp / "data.txt"
        f.write_text("hello", encoding="utf-8")
        node = FileInputNode({"path": str(f)})
        out = node.execute({}, node.params, {})
        self.assertEqual(out["file"].data_kind, "file")
        self.assertEqual(out["file"].data, str(f.resolve()))

    def test_file_input_folder(self) -> None:
        d = self.tmp / "imgs"
        d.mkdir()
        (d / "a.png").write_bytes(b"x")
        node = FileInputNode({"path": str(d)})
        out = node.execute({}, node.params, {})
        self.assertEqual(out["folder"].data_kind, "folder")
        self.assertEqual(out["dataset"].data_kind, "dataset")
        self.assertEqual(out["dataset"].data["count"], 1)

    def test_dataset_scan(self) -> None:
        d = self.tmp / "ds"
        d.mkdir()
        for i in range(3):
            (d / f"{i}.jpg").write_bytes(b"x")
        node = DatasetNode({"path": str(d), "split": "train"})
        out = node.execute({}, node.params, {})
        self.assertEqual(out["dataset"].data_kind, "dataset")
        self.assertEqual(out["dataset"].data["count"], 3)
        self.assertEqual(out["dataset"].data["split"], "train")

    def test_checkpoint_roundtrip_safetensors(self) -> None:
        path = self.tmp / "ckpt.safetensors"
        save = CheckpointSaveNode({"path": str(path), "format": "auto"})
        save.execute({"model": _make_model()}, save.params, {})
        self.assertTrue(path.is_file())

        # structure rebuilt from the dataset-style module is not needed here:
        # use template input instead
        template = _make_model()
        load = CheckpointLoadNode({"path": str(path), "device": "cpu"})
        out = load.execute({"template": template}, load.params, {})
        m = out["model"]
        self.assertEqual(tuple(m(torch.randn(3, 4)).shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
