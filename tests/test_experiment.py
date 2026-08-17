"""Experiment records: reproducibility capture."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from entropia_riko.project import hardware_info, list_experiments, record_experiment


class TestExperiment(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def test_record_creates_structure(self) -> None:
        exp = record_experiment(
            self.tmp,
            workflow={"nodes": [], "edges": []},
            parameters={"lr": 1e-3},
            metrics={"loss": 0.5},
            seed=42,
        )
        self.assertEqual(exp.name, "experiment_001")
        self.assertTrue((exp / "workflow.json").is_file())
        self.assertTrue((exp / "parameters.json").is_file())
        self.assertTrue((exp / "metrics.json").is_file())
        self.assertTrue((exp / "metadata.json").is_file())
        self.assertTrue((exp / "outputs").is_dir())
        self.assertTrue((exp / "logs").is_dir())

    def test_metrics_roundtrip(self) -> None:
        record_experiment(self.tmp, {}, {}, {"loss": 0.25, "acc": 0.9})
        exps = list_experiments(self.tmp)
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0]["metrics"], {"loss": 0.25, "acc": 0.9})

    def test_increments(self) -> None:
        record_experiment(self.tmp, {}, {}, {})
        exp2 = record_experiment(self.tmp, {}, {}, {})
        self.assertEqual(exp2.name, "experiment_002")

    def test_metadata_records_hardware(self) -> None:
        exp = record_experiment(self.tmp, {}, {}, {}, seed=7)
        meta = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["seed"], 7)
        self.assertIn("python", meta["hardware"])

    def test_hardware_info(self) -> None:
        info = hardware_info()
        self.assertIn("python", info)
        self.assertIn("platform", info)


if __name__ == "__main__":
    unittest.main()
