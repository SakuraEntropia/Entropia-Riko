"""Stage 6 tests: broad torch API nodes + model loader/inference.

torch 未安装时跳过。运行：
    .venv/bin/python -m unittest discover -s tests -t .
"""
import os
import tempfile
import unittest

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

import src.nodes  # noqa: F401  触发全部节点注册
from src.core.tensor import TensorValue
from src.runtime.registry import default_registry
from src.core.document import GraphDocument, NodeModel, EdgeModel
from src.runtime.executor import execute


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装")
class TestApiNodes(unittest.TestCase):
    def test_many_registered(self):
        reg = default_registry()
        for t in [
            "abs", "exp", "sqrt", "sigmoid", "tanh", "gelu", "silu",
            "sub", "div", "matmul", "pow", "maximum",
            "sum", "mean", "std", "norm", "argmax", "cumsum",
            "reshape", "transpose", "permute", "flatten", "squeeze", "concat", "stack",
            "softmax", "log_softmax",
            "conv2d", "maxpool2d", "avgpool2d", "embedding", "dropout",
            "batchnorm1d", "layernorm",
            "model_loader", "inference",
        ]:
            self.assertIn(t, reg, f"节点类型 '{t}' 未注册")

    def test_unary_abs(self):
        node = default_registry().get("abs")()
        out = node.execute({"x": TensorValue([-1.0, 2.0, -3.0])}, node.params, {})
        self.assertEqual(out["result"].to_list(), [1.0, 2.0, 3.0])

    def test_unary_sigmoid(self):
        node = default_registry().get("sigmoid")()
        out = node.execute({"x": TensorValue(0.0)}, node.params, {})
        self.assertAlmostEqual(out["result"].item(), 0.5, places=5)

    def test_binary_sub(self):
        node = default_registry().get("sub")()
        out = node.execute(
            {"left": TensorValue(5.0), "right": TensorValue(3.0)}, node.params, {}
        )
        self.assertEqual(out["result"].item(), 2.0)

    def test_matmul(self):
        node = default_registry().get("matmul")()
        out = node.execute(
            {"left": TensorValue([[1, 2], [3, 4]]), "right": TensorValue([[1, 0], [0, 1]])},
            node.params, {},
        )
        self.assertEqual(out["result"].to_list(), [[1, 2], [3, 4]])

    def test_reduce_sum(self):
        node = default_registry().get("sum")()
        out = node.execute({"x": TensorValue([1, 2, 3, 4])}, node.params, {})
        self.assertEqual(out["result"].item(), 10.0)

    def test_reduce_mean_dim(self):
        node = default_registry().get("mean")()
        out = node.execute(
            {"x": TensorValue([[1, 2], [3, 4]])}, node.params | {"dim": 1}, {}
        )
        self.assertEqual(out["result"].to_list(), [1.5, 3.5])

    def test_argmax(self):
        node = default_registry().get("argmax")()
        out = node.execute({"x": TensorValue([1.0, 5.0, 3.0])}, node.params, {})
        self.assertEqual(out["result"].item(), 1)

    def test_reshape(self):
        node = default_registry().get("reshape")({"shape": [2, 2]})
        out = node.execute({"x": TensorValue([1, 2, 3, 4])}, node.params, {})
        self.assertEqual(out["result"].shape, (2, 2))

    def test_transpose(self):
        node = default_registry().get("transpose")({"dim0": 0, "dim1": 1})
        out = node.execute({"x": TensorValue([[1, 2], [3, 4]])}, node.params, {})
        self.assertEqual(out["result"].to_list(), [[1, 3], [2, 4]])

    def test_concat(self):
        node = default_registry().get("concat")({"dim": 0})
        out = node.execute(
            {"left": TensorValue([1, 2]), "right": TensorValue([3, 4])}, node.params, {}
        )
        self.assertEqual(out["result"].to_list(), [1, 2, 3, 4])

    def test_softmax_sums_to_one(self):
        node = default_registry().get("softmax")()
        out = node.execute({"x": TensorValue([1.0, 2.0, 3.0])}, node.params, {})
        self.assertAlmostEqual(sum(out["result"].to_list()), 1.0, places=5)

    def test_layernorm(self):
        node = default_registry().get("layernorm")({"normalized_shape": [3]})
        out = node.execute({"x": TensorValue([[1.0, 2.0, 3.0]])}, node.params, {})
        self.assertEqual(out["result"].shape, (1, 3))

    def test_conv2d_shape(self):
        node = default_registry().get("conv2d")({
            "in_channels": 1, "out_channels": 2, "kernel_size": 3, "padding": 1,
        })
        out = node.execute({"x": TensorValue([[[[1.0, 2.0], [3.0, 4.0]]]])}, node.params, {})
        self.assertEqual(out["result"].shape, (1, 2, 2, 2))

    def test_api_graph(self):
        # constant([1,-2,3]) -> abs -> sum -> result 6
        doc = GraphDocument()
        doc.add_node(NodeModel(id="in", type_name="constant", parameters={"value": [1.0, -2.0, 3.0]}))
        doc.add_node(NodeModel(id="a", type_name="abs"))
        doc.add_node(NodeModel(id="s", type_name="sum"))
        doc.add_edge(EdgeModel(id="e1", source_node="in", source_port="value", target_node="a", target_port="x"))
        doc.add_edge(EdgeModel(id="e2", source_node="a", source_port="result", target_node="s", target_port="x"))
        out = execute(doc)
        self.assertEqual(out["s"]["result"].item(), 6.0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装")
class TestModelLoader(unittest.TestCase):
    def test_loader_inference_e2e(self):
        model = nn.Linear(3, 2)
        model.weight.data = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32)
        model.bias.data = torch.tensor([0.0, 0.0], dtype=torch.float32)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(model, path)
            doc = GraphDocument()
            doc.add_node(NodeModel(id="m", type_name="model_loader", parameters={"path": path}))
            doc.add_node(NodeModel(id="in", type_name="constant", parameters={"value": [1.0, 2.0, 3.0]}))
            doc.add_node(NodeModel(id="inf", type_name="inference"))
            doc.add_edge(EdgeModel(id="e1", source_node="m", source_port="model", target_node="inf", target_port="model"))
            doc.add_edge(EdgeModel(id="e2", source_node="in", source_port="value", target_node="inf", target_port="x"))
            out = execute(doc)
            # linear(ones weight): [1+2+3, 1+2+3] = [6, 6]
            self.assertEqual(out["inf"]["output"].to_list(), [6.0, 6.0])
        finally:
            os.unlink(path)

    def test_loader_missing_file(self):
        node = default_registry().get("model_loader")({"path": "/nonexistent/path.pt"})
        with self.assertRaises(ValueError):
            node.execute({}, node.params, {})


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装")
class TestCodegen(unittest.TestCase):
    def test_export_add(self):
        from src.runtime.codegen import export_python
        doc = GraphDocument()
        doc.add_node(NodeModel(id="a", type_name="constant", parameters={"value": 2.0}))
        doc.add_node(NodeModel(id="b", type_name="constant", parameters={"value": 3.0}))
        doc.add_node(NodeModel(id="s", type_name="add"))
        doc.add_edge(EdgeModel(id="e1", source_node="a", source_port="value", target_node="s", target_port="left"))
        doc.add_edge(EdgeModel(id="e2", source_node="b", source_port="value", target_node="s", target_port="right"))
        code = export_python(doc)
        self.assertIn("torch.tensor", code)
        self.assertIn("+", code)
        self.assertIn("import torch", code)

    def test_export_transformer(self):
        from src.runtime.codegen import export_python
        doc = GraphDocument()
        doc.add_node(NodeModel(id="x", type_name="constant", parameters={"value": [[1.0] * 16] * 3}))
        doc.add_node(NodeModel(id="enc", type_name="transformer_encoder", parameters={"d_model": 16, "nhead": 4, "num_layers": 1}))
        doc.add_edge(EdgeModel(id="e1", source_node="x", source_port="value", target_node="enc", target_port="x"))
        code = export_python(doc)
        self.assertIn("TransformerEncoder", code)
        self.assertIn("import torch.nn as nn", code)

    def test_export_creation(self):
        from src.runtime.codegen import export_python
        doc = GraphDocument()
        doc.add_node(NodeModel(id="z", type_name="zeros", parameters={"shape": [3, 4]}))
        code = export_python(doc)
        self.assertIn("torch.zeros", code)
        self.assertIn("(3, 4)", code)


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装")
class TestSubgraph(unittest.TestCase):
    def test_graph_input_output_direct(self):
        cls_in = default_registry().get("graph_input")
        cls_out = default_registry().get("graph_output")
        node_in = cls_in({"name": "x"})
        node_out = cls_out({"name": "y"})
        ctx = {"graph_inputs": {"x": TensorValue(42.0)}}
        r1 = node_in.execute({}, node_in.params, ctx)
        r2 = node_out.execute({"value": r1["value"]}, node_out.params, ctx)
        self.assertEqual(r2, {})
        self.assertEqual(ctx["graph_outputs"]["y"].item(), 42.0)

    def test_graph_reference_e2e(self):
        import os
        riko_path = os.path.join(
            os.path.dirname(__file__), "..", "examples", "transformer_pipeline.riko"
        )
        doc = GraphDocument()
        doc.add_node(NodeModel(id="tok", type_name="constant", parameters={"value": [[1, 2, 3, 4, 5]]}))
        doc.add_node(NodeModel(id="ref", type_name="graph_reference", parameters={"file": riko_path}))
        doc.add_edge(EdgeModel(id="e1", source_node="tok", source_port="value", target_node="ref", target_port="input"))
        out = execute(doc)
        # tokens (1,5) -> embedding (1,5,16) -> transformer (1,5,16) -> linear (1,5,5)
        self.assertEqual(out["ref"]["output"].shape, (1, 5, 5))

    def test_subgraph_registered(self):
        self.assertIn("graph_input", default_registry())
        self.assertIn("graph_output", default_registry())
        self.assertIn("graph_reference", default_registry())


if __name__ == "__main__":
    unittest.main()
