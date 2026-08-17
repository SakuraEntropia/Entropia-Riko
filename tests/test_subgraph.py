"""Tests for the import node, subgraph resolution, and clean codegen export.

运行：.venv/bin/python -m unittest discover -s tests -t .
"""
import json
import unittest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import entropia_riko.nodes  # noqa: F401  触发全部节点注册（含 import）
from entropia_riko.core.document import GraphDocument, NodeModel, EdgeModel
from entropia_riko.core.tensor import TensorValue
from entropia_riko.runtime.registry import default_registry
from entropia_riko.runtime.subgraph import resolve_graph_file, PROJECT_ROOT
from entropia_riko.runtime.codegen import export_python


class TestSubgraphResolution(unittest.TestCase):
    def test_import_registered(self):
        self.assertIn("import", default_registry())

    def test_resolve_module_name(self):
        p = resolve_graph_file("mlp")
        self.assertIsNotNone(p)
        self.assertTrue(str(p).endswith("mlp.riko"))

    def test_resolve_relative_path(self):
        p = resolve_graph_file("examples/models/mlp.riko")
        self.assertIsNotNone(p)

    def test_resolve_missing(self):
        self.assertIsNone(resolve_graph_file("does_not_exist_xyz"))

    def test_text_kind(self):
        tv = TensorValue("hello", kind="text")
        self.assertEqual(tv.data_kind, "text")
        self.assertEqual(tv.shape, ())

    def test_json_kind(self):
        tv = TensorValue({"a": 1}, kind="json")
        self.assertEqual(tv.data_kind, "json")


class TestHfNodes(unittest.TestCase):
    def test_hf_nodes_registered(self):
        for t in ("diffusers_text2img", "transformers_pipeline", "transformers_embedding"):
            self.assertIn(t, default_registry())

    def test_hf_codegen_compiles(self):
        from entropia_riko.runtime.codegen import export_python
        doc = GraphDocument.from_dict({
            "version": "1.0",
            "nodes": [
                {"id": "g", "type_name": "diffusers_text2img",
                 "parameters": {"model_id": "x", "prompt": "a cat"}},
                {"id": "p", "type_name": "transformers_pipeline",
                 "parameters": {"task": "sentiment-analysis", "model_id": "y", "text": "hi"}},
            ],
            "edges": [], "settings": {},
        })
        code = export_python(doc)
        compile(code, "<generated>", "exec")
        self.assertIn("DiffusionPipeline", code)
        self.assertIn("from transformers import pipeline", code)


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装")
class TestWrangle(unittest.TestCase):
    def test_wrangle_executes(self):
        w = default_registry().get("wrangle")({"code": "result = x * 2"})
        out = w.execute({"x": TensorValue([1.0, 2.0])}, w.params, {})
        self.assertEqual(out["result"].to_list(), [2.0, 4.0])

    def test_wrangle_codegen_compiles(self):
        from entropia_riko.runtime.codegen import export_python
        doc = GraphDocument()
        doc.add_node(NodeModel(id="in", type_name="constant", parameters={"value": [1.0, 2.0]}))
        doc.add_node(NodeModel(id="w", type_name="wrangle", parameters={"code": "result = x + 1"}))
        doc.add_edge(EdgeModel(id="e1", source_node="in", source_port="value", target_node="w", target_port="x"))
        code = export_python(doc)
        compile(code, "<generated>", "exec")
        self.assertIn("result = x + 1", code)


class TestKerasExport(unittest.TestCase):
    """TF/Keras nodes register + export to a tf.keras.Model script (no TF needed to generate)."""

    def test_keras_nodes_registered(self):
        for t in ("keras_dense", "keras_conv2d", "keras_relu", "keras_softmax",
                  "keras_flatten", "tf_matmul", "tf_reduce", "tf_reshape"):
            self.assertIn(t, default_registry())

    def test_export_keras_compiles(self):
        from entropia_riko.runtime.codegen_tf import export_keras
        doc = GraphDocument.from_dict({
            "version": "0.1",
            "nodes": [
                {"id": "gin", "type_name": "graph_input", "parameters": {"name": "input"}},
                {"id": "d1", "type_name": "keras_dense", "parameters": {"units": 64}},
                {"id": "r1", "type_name": "keras_relu"},
                {"id": "d2", "type_name": "keras_dense", "parameters": {"units": 10}},
                {"id": "sm", "type_name": "keras_softmax"},
                {"id": "gout", "type_name": "graph_output", "parameters": {"name": "output"}},
            ],
            "edges": [
                {"id": "e0", "source_node": "gin", "source_port": "value", "target_node": "d1", "target_port": "x"},
                {"id": "e1", "source_node": "d1", "source_port": "output", "target_node": "r1", "target_port": "x"},
                {"id": "e2", "source_node": "r1", "source_port": "result", "target_node": "d2", "target_port": "x"},
                {"id": "e3", "source_node": "d2", "source_port": "output", "target_node": "sm", "target_port": "x"},
                {"id": "e4", "source_node": "sm", "source_port": "result", "target_node": "gout", "target_port": "value"},
            ],
            "settings": {},
        })
        code = export_keras(doc)
        compile(code, "<generated>", "exec")
        self.assertIn("tf.keras.Model", code)
        self.assertIn("tf.keras.layers.Dense", code)
        self.assertIn("tf.nn.relu", code)

    def test_export_torch_graph_to_keras_is_clean(self):
        """A torch-style graph (conv/relu/pool/linear) exports clean Keras, no None stubs."""
        from entropia_riko.runtime.codegen_tf import export_keras
        doc = self._load_cnn()
        code = export_keras(doc)
        compile(code, "<generated>", "exec")
        self.assertIn("tf.keras.layers.Conv2D", code)
        self.assertIn("tf.keras.layers.MaxPooling2D", code)
        self.assertIn("tf.keras.layers.Dense", code)
        self.assertIn("tf.nn.relu", code)
        self.assertIn("tf.keras.layers.Flatten", code)
        self.assertNotIn("None  #", code)
        self.assertNotIn("未支持", code)

    @staticmethod
    def _load_cnn():
        p = PROJECT_ROOT / "examples" / "models" / "cnn.riko"
        return GraphDocument.from_dict(json.loads(p.read_text(encoding="utf-8")))


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装")
class TestImportExecution(unittest.TestCase):
    def test_import_executes_mlp(self):
        from entropia_riko.runtime.executor import execute
        doc = GraphDocument()
        doc.add_node(NodeModel(id="in", type_name="constant", parameters={"value": [[1.0] * 8]}))
        doc.add_node(NodeModel(id="imp", type_name="import", parameters={"module": "mlp"}))
        doc.add_edge(EdgeModel(id="e1", source_node="in", source_port="value", target_node="imp", target_port="input"))
        out = execute(doc)
        self.assertEqual(out["imp"]["output"].shape, (1, 4))

    def test_classifier_e2e(self):
        from entropia_riko.runtime.executor import execute
        doc = GraphDocument()
        doc.add_node(NodeModel(id="in", type_name="constant", parameters={"value": [[1.0] * 8]}))
        doc.add_node(NodeModel(id="imp", type_name="import", parameters={"module": "classifier"}))
        doc.add_edge(EdgeModel(id="e1", source_node="in", source_port="value", target_node="imp", target_port="input"))
        out = execute(doc)
        self.assertEqual(out["imp"]["output"].shape, (1, 4))
        # softmax over last dim -> sums to 1
        probs = out["imp"]["output"].to_list()
        self.assertAlmostEqual(sum(probs[0]), 1.0, places=5)


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装")
class TestCodegenClean(unittest.TestCase):
    def _load(self, name):
        p = PROJECT_ROOT / "examples" / "models" / f"{name}.riko"
        return GraphDocument.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def test_export_classifier_is_clean(self):
        code = export_python(self._load("classifier"))
        # structure
        self.assertIn("class GraphModel(nn.Module):", code)
        self.assertIn("def __init__(self):", code)
        self.assertIn("super().__init__()", code)
        self.assertIn("def forward(self,", code)
        # full graph inlined (no import stub)
        self.assertIn("nn.Linear(8, 16", code)
        self.assertIn("nn.Linear(32, 4", code)
        self.assertIn("torch.relu", code)
        self.assertIn("F.softmax", code)
        self.assertNotIn("class Mlp", code)
        # no messy artifacts
        self.assertNotIn("None  #", code)
        self.assertNotIn("unknown layer type", code)
        self.assertNotIn(" = None", code)
        # valid Python
        compile(code, "<generated>", "exec")

    def test_export_import_inlines_nested_module(self):
        """An `import` node still inlines its target as a nested nn.Module."""
        doc = GraphDocument()
        doc.add_node(NodeModel(id="gin", type_name="graph_input", parameters={"name": "input"}))
        doc.add_node(NodeModel(id="block", type_name="import", parameters={"module": "mlp"}))
        doc.add_node(NodeModel(id="gout", type_name="graph_output", parameters={"name": "output"}))
        doc.add_edge(EdgeModel(id="e0", source_node="gin", source_port="value", target_node="block", target_port="input"))
        doc.add_edge(EdgeModel(id="e1", source_node="block", source_port="output", target_node="gout", target_port="value"))
        code = export_python(doc)
        compile(code, "<generated>", "exec")
        self.assertIn("class Mlp(nn.Module):", code)
        self.assertIn("self.block_output = Mlp()", code)

    def test_export_mlp_clean(self):
        code = export_python(self._load("mlp"))
        compile(code, "<generated>", "exec")
        self.assertIn("nn.Linear(8, 16", code)
        self.assertIn("torch.relu", code)

    def test_export_add_clean(self):
        doc = GraphDocument()
        doc.add_node(NodeModel(id="a", type_name="constant", parameters={"value": 2.0}))
        doc.add_node(NodeModel(id="b", type_name="constant", parameters={"value": 3.0}))
        doc.add_node(NodeModel(id="s", type_name="add"))
        doc.add_edge(EdgeModel(id="e1", source_node="a", source_port="value", target_node="s", target_port="left"))
        doc.add_edge(EdgeModel(id="e2", source_node="b", source_port="value", target_node="s", target_port="right"))
        code = export_python(doc)
        compile(code, "<generated>", "exec")
        self.assertIn("s = a + b", code)
        self.assertIn("a = torch.tensor(2.0", code)
        self.assertNotIn(" = None", code)

    def test_export_creation_clean(self):
        doc = GraphDocument()
        doc.add_node(NodeModel(id="z", type_name="zeros", parameters={"shape": [3, 4]}))
        code = export_python(doc)
        compile(code, "<generated>", "exec")
        self.assertIn("torch.zeros((3, 4))", code)

    def test_export_advanced_nodes_compile(self):
        """Advanced nodes (attention / RMSNorm / einsum / interpolate / ...) codegen."""
        doc = GraphDocument.from_dict({
            "version": "0.1",
            "nodes": [
                {"id": "pe", "type_name": "positional_encoding", "parameters": {"n": 4, "dim": 8}},
                {"id": "x", "type_name": "zeros", "parameters": {"shape": [4, 8]}},
                {"id": "s", "type_name": "add"},
                {"id": "n", "type_name": "rmsnorm", "parameters": {"dim": 8}},
                {"id": "a", "type_name": "multihead_attention", "parameters": {"embed_dim": 8, "num_heads": 2}},
                {"id": "o", "type_name": "amax"},
            ],
            "edges": [
                {"id": "e1", "source_node": "x", "source_port": "result", "target_node": "s", "target_port": "left"},
                {"id": "e2", "source_node": "pe", "source_port": "result", "target_node": "s", "target_port": "right"},
                {"id": "e3", "source_node": "s", "source_port": "result", "target_node": "n", "target_port": "x"},
                {"id": "e4", "source_node": "n", "source_port": "output", "target_node": "a", "target_port": "x"},
                {"id": "e5", "source_node": "a", "source_port": "output", "target_node": "o", "target_port": "x"},
            ],
            "settings": {},
        })
        code = export_python(doc)
        compile(code, "<generated>", "exec")
        self.assertIn("class RMSNorm", code)
        self.assertIn("nn.MultiheadAttention", code)
        self.assertIn("_sincos_1d", code)
        self.assertIn("torch.amax", code)

    def test_export_runs_end_to_end(self):
        """The generated classifier module actually runs and matches shapes."""
        code = export_python(self._load("classifier"))
        ns = {}
        exec(code, ns)  # noqa: S102 - generated code under test
        model = ns["GraphModel"]()
        x = torch.randn(2, 8)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 4))
        # softmax rows sum to 1
        self.assertTrue(torch.allclose(y.sum(dim=-1), torch.ones(2), atol=1e-5))

    def test_export_mnist_cnn_is_trainable(self):
        """Self-contained data+loss graph exports to a trainable script."""
        code = export_python(self._load("mnist_cnn"))
        compile(code, "<generated>", "exec")
        self.assertIn("def train(", code)
        self.assertIn("loss.backward()", code)
        self.assertIn("optimizer.step()", code)
        self.assertIn("def forward(self):", code)  # data loaded internally
        self.assertIn("_load_data()", code)
        # no broken placeholders in the forward path
        self.assertNotIn("self.p1(_)", code)
        self.assertNotIn("self.p2(_)", code)
        self.assertNotIn("None  # MNIST", code)

    def test_export_train_runs(self):
        """The generated train() actually runs a few optimizer steps."""
        doc = GraphDocument.from_dict({
            "version": "0.1",
            "nodes": [
                {"id": "data", "type_name": "dataloader",
                 "parameters": {"batch_size": 8, "channels": 1, "height": 28, "width": 28}},
                {"id": "f", "type_name": "flatten",
                 "parameters": {"start_dim": 1, "end_dim": -1}},
                {"id": "l", "type_name": "linear",
                 "parameters": {"in_features": 784, "out_features": 10}},
                {"id": "loss", "type_name": "cross_entropy_loss", "parameters": {}},
                {"id": "lbl", "type_name": "constant",
                 "parameters": {"value": [0, 1, 2, 3, 4, 5, 6, 7]}},
            ],
            "edges": [
                {"id": "e0", "source_node": "data", "source_port": "data",
                 "target_node": "f", "target_port": "x"},
                {"id": "e1", "source_node": "f", "source_port": "result",
                 "target_node": "l", "target_port": "x"},
                {"id": "e2", "source_node": "l", "source_port": "output",
                 "target_node": "loss", "target_port": "pred"},
                {"id": "e3", "source_node": "lbl", "source_port": "value",
                 "target_node": "loss", "target_port": "target"},
            ],
            "settings": {},
        })
        code = export_python(doc)
        ns = {}
        exec(code, ns)  # noqa: S102 - generated code under test
        losses = ns["train"](ns["GraphModel"](), steps=3, lr=1e-2)
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(isinstance(x, float) for x in losses))


if __name__ == "__main__":
    unittest.main()
