"""Strong-typed port system: compatibility rules + connection validation."""

from __future__ import annotations

import unittest

import entropia_riko.nodes  # noqa: F401  触发节点注册
from entropia_riko.core.document import GraphDocument
from entropia_riko.core.types import DATA_KINDS, is_compatible, normalize
from entropia_riko.nodes.base import BaseNode, NodeInput, NodeOutput
from entropia_riko.runtime.executor import validate
from entropia_riko.runtime.registry import Registry, register


class TestTypeCompatibility(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertTrue(is_compatible("tensor", "tensor"))
        self.assertTrue(is_compatible("model", "model"))

    def test_subtype_widening(self) -> None:
        self.assertTrue(is_compatible("image_tensor", "tensor"))
        self.assertTrue(is_compatible("mask", "image_tensor"))
        self.assertTrue(is_compatible("mask", "tensor"))
        self.assertTrue(is_compatible("checkpoint", "file"))

    def test_incompatible(self) -> None:
        self.assertFalse(is_compatible("text", "audio"))
        self.assertFalse(is_compatible("tensor", "image_tensor"))  # no narrowing
        self.assertFalse(is_compatible("model", "image_tensor"))

    def test_unknown_is_wildcard(self) -> None:
        self.assertTrue(is_compatible("unknown", "tensor"))
        self.assertTrue(is_compatible("tensor", "unknown"))

    def test_normalize(self) -> None:
        self.assertEqual(normalize("tensor"), "tensor")
        self.assertEqual(normalize("not-a-kind"), "unknown")

    def test_kinds_include_professional_types(self) -> None:
        for k in ("dataset", "checkpoint", "model", "embedding", "latent", "audio", "video", "config"):
            self.assertIn(k, DATA_KINDS)


class TestConnectionValidation(unittest.TestCase):
    def _registry(self) -> Registry:
        reg = Registry()

        @register("src_text", reg)
        class _SrcText(BaseNode):
            type_name = "src_text"
            inputs = []
            outputs = [NodeOutput("out", data_kind="text")]

        @register("dst_audio", reg)
        class _DstAudio(BaseNode):
            type_name = "dst_audio"
            inputs = [NodeInput("in", data_kind="audio", required=True)]
            outputs = []

        return reg

    def test_rejects_incompatible_connection(self) -> None:
        reg = self._registry()
        doc = GraphDocument.from_dict({
            "version": "1.0",
            "nodes": [
                {"id": "a", "type_name": "src_text", "parameters": {}, "inputs": [], "outputs": []},
                {"id": "b", "type_name": "dst_audio", "parameters": {}, "inputs": [], "outputs": []},
            ],
            "edges": [
                {"id": "e0", "source_node": "a", "source_port": "out",
                 "target_node": "b", "target_port": "in"},
            ],
        })
        errors = validate(doc, reg)
        self.assertTrue(any("类型不兼容" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
