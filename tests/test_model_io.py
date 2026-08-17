"""Model save/load nodes: train-to-model → load-from-model → inference."""

from __future__ import annotations

import os
import tempfile
import unittest

import torch
import torch.nn as nn

import entropia_riko.nodes  # noqa: F401  触发全部节点注册
from entropia_riko.nodes.torch_ops.model_loader import ModelLoaderNode, SaveModelNode
from entropia_riko.runtime.registry import default_registry


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


class TestModelIO(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = default_registry()
        self.model = _make_model()
        self.x = torch.randn(3, 4)
        self.tmp = tempfile.mkdtemp()

    def test_nodes_registered(self) -> None:
        self.assertIn("save_model", self.reg.list())
        self.assertIn("model_loader", self.reg.list())

    def test_save_load_safetensors_with_template(self) -> None:
        path = os.path.join(self.tmp, "model.safetensors")
        save = SaveModelNode({"path": path, "format": "auto"})
        save.execute({"model": self.model}, save.params, {})
        self.assertTrue(os.path.exists(path))

        template = _make_model()
        load = ModelLoaderNode({"path": path, "device": "cpu"})
        out = load.execute({"template": template}, load.params, {})
        m = out["model"]
        self.assertTrue(hasattr(m, "state_dict"))
        self.assertEqual(tuple(m(self.x).shape), (3, 2))

    def test_save_load_pt_state_dict(self) -> None:
        path = os.path.join(self.tmp, "model.pt")
        save = SaveModelNode({"path": path, "format": "auto"})
        save.execute({"model": self.model}, save.params, {})
        template = _make_model()
        load = ModelLoaderNode({"path": path, "device": "cpu"})
        out = load.execute({"template": template}, load.params, {})
        m = out["model"]
        self.assertEqual(tuple(m(self.x).shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
