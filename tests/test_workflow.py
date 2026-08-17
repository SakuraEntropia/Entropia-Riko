"""Workflow metadata and dependency ordering."""

from __future__ import annotations

import unittest

from entropia_riko.core.document import GraphDocument
from entropia_riko.project import (
    WorkflowMetadata,
    attach_metadata,
    dependency_order,
    extract_metadata,
)


class TestWorkflowMetadata(unittest.TestCase):
    def test_roundtrip(self) -> None:
        doc = GraphDocument()
        attach_metadata(doc, WorkflowMetadata(
            name="train", version="2", category="training",
            dependencies=["dataset_prepare"], input_types=["dataset"],
            output_types=["checkpoint"],
        ))
        meta = extract_metadata(doc)
        self.assertEqual(meta.name, "train")
        self.assertEqual(meta.version, "2")
        self.assertEqual(meta.category, "training")
        self.assertEqual(meta.dependencies, ["dataset_prepare"])

    def test_category_normalized(self) -> None:
        m = WorkflowMetadata(name="x", category="bogus")
        self.assertEqual(m.category, "utility")

    def test_dependency_order(self) -> None:
        wfs = {
            "inference": WorkflowMetadata(name="inference", dependencies=["training"]),
            "training": WorkflowMetadata(name="training", dependencies=["dataset"]),
            "dataset": WorkflowMetadata(name="dataset"),
            "export": WorkflowMetadata(name="export", dependencies=["inference"]),
        }
        order = dependency_order(wfs)
        self.assertEqual(len(order), 4)
        # dataset before training before inference before export
        self.assertLess(order.index("dataset"), order.index("training"))
        self.assertLess(order.index("training"), order.index("inference"))
        self.assertLess(order.index("inference"), order.index("export"))

    def test_cycle_tolerated(self) -> None:
        wfs = {
            "a": WorkflowMetadata(name="a", dependencies=["b"]),
            "b": WorkflowMetadata(name="b", dependencies=["a"]),
        }
        order = dependency_order(wfs)
        self.assertEqual(sorted(order), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
