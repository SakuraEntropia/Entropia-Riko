"""Stage 3 backend tests: device detection, tensor conversion, torch ops.

torch 未安装时跳过。运行：
    python -m unittest discover -s tests -t .
    # 或用 venv: .venv/bin/python -m unittest discover -s tests -t .
"""
import unittest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import entropia_riko.nodes  # noqa: F401  触发节点注册（含 torch_ops）
from entropia_riko.core.tensor import TensorValue
from entropia_riko.backend.device import resolve_device
from entropia_riko.backend.converter import to_torch, from_torch
from entropia_riko.runtime.registry import default_registry


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装，跳过 backend 测试")
class TestDevice(unittest.TestCase):
    def test_cpu_always(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_auto_returns_valid(self):
        dev = resolve_device("auto")
        self.assertIn(dev.type, ("cpu", "cuda", "mps"))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            resolve_device("tpu")

    def test_cuda_unavailable_raises(self):
        if torch.cuda.is_available():
            self.assertEqual(resolve_device("cuda").type, "cuda")
        else:
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装，跳过 backend 测试")
class TestConverter(unittest.TestCase):
    def test_roundtrip_scalar(self):
        tv = TensorValue(3.5, dtype="float32")
        back = from_torch(to_torch(tv))
        self.assertEqual(back.shape, ())
        self.assertEqual(back.item(), 3.5)

    def test_roundtrip_vector(self):
        tv = TensorValue.from_value([1, 2, 3], dtype="int32")
        back = from_torch(to_torch(tv))
        self.assertEqual(back.shape, (3,))
        self.assertEqual(back.to_list(), [1, 2, 3])
        self.assertEqual(back.dtype, "int32")

    def test_torch_dtype(self):
        tv = TensorValue([1.0, 2.0], dtype="float32")
        t = to_torch(tv)
        self.assertEqual(t.dtype, torch.float32)
        self.assertEqual(tuple(t.shape), (2,))


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装，跳过 backend 测试")
class TestTorchOps(unittest.TestCase):
    def test_torch_add_registered(self):
        self.assertIn("torch_add", default_registry())
        self.assertIn("torch_multiply", default_registry())

    def test_torch_add_scalar(self):
        cls = default_registry().get("torch_add")
        node = cls()
        out = node.execute(
            {"left": TensorValue(2.0), "right": TensorValue(3.0)},
            node.params, {},
        )
        self.assertEqual(out["result"].item(), 5.0)
        self.assertEqual(out["result"].metadata["backend"], "torch")

    def test_torch_add_broadcast(self):
        cls = default_registry().get("torch_add")
        node = cls()
        out = node.execute(
            {"left": TensorValue([1, 2, 3]), "right": TensorValue(10)},
            node.params, {},
        )
        self.assertEqual(out["result"].to_list(), [11, 12, 13])

    def test_torch_multiply(self):
        cls = default_registry().get("torch_multiply")
        node = cls()
        out = node.execute(
            {"left": TensorValue(4.0), "right": TensorValue(5.0)},
            node.params, {},
        )
        self.assertEqual(out["result"].item(), 20.0)

    def test_torch_add_via_executor(self):
        from entropia_riko.core.document import GraphDocument, NodeModel, EdgeModel
        from entropia_riko.runtime.executor import execute

        doc = GraphDocument()
        doc.add_node(NodeModel(id="a", type_name="constant", parameters={"value": 2.0}))
        doc.add_node(NodeModel(id="b", type_name="constant", parameters={"value": 3.0}))
        doc.add_node(NodeModel(id="sum", type_name="torch_add"))
        doc.add_edge(EdgeModel(id="e1", source_node="a", source_port="value", target_node="sum", target_port="left"))
        doc.add_edge(EdgeModel(id="e2", source_node="b", source_port="value", target_node="sum", target_port="right"))
        out = execute(doc)
        self.assertEqual(out["sum"]["result"].item(), 5.0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch 未安装，跳过 backend 测试")
class TestNeuralNodes(unittest.TestCase):
    def test_linear_shape(self):
        cls = default_registry().get("linear")
        node = cls({"in_features": 3, "out_features": 2})
        out = node.execute({"x": TensorValue([1.0, 2.0, 3.0])}, node.params, {})
        self.assertEqual(out["output"].shape, (2,))

    def test_linear_with_weight(self):
        cls = default_registry().get("linear")
        node = cls({
            "in_features": 3, "out_features": 2,
            "weight": [[1, 1, 1], [1, 1, 1]], "bias": [0.0, 0.0],
        })
        out = node.execute({"x": TensorValue([1.0, 2.0, 3.0])}, node.params, {})
        self.assertEqual(out["output"].to_list(), [6.0, 6.0])

    def test_linear_bad_weight_shape(self):
        cls = default_registry().get("linear")
        node = cls({"in_features": 3, "out_features": 2, "weight": [[1, 1]]})
        with self.assertRaises(ValueError):
            node.execute({"x": TensorValue([1.0, 2.0, 3.0])}, node.params, {})

    def test_relu(self):
        cls = default_registry().get("relu")
        node = cls()
        out = node.execute({"x": TensorValue([-1.0, 2.0, -3.0])}, node.params, {})
        self.assertEqual(out["output"].to_list(), [0.0, 2.0, 0.0])

    def test_linear_relu_graph(self):
        from entropia_riko.core.document import GraphDocument, NodeModel, EdgeModel
        from entropia_riko.runtime.executor import execute

        doc = GraphDocument()
        doc.add_node(NodeModel(id="in", type_name="constant", parameters={"value": [1.0, -2.0, 3.0]}))
        doc.add_node(NodeModel(id="lin", type_name="linear", parameters={
            "in_features": 3, "out_features": 3,
            "weight": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "bias": [0, 0, 0],
        }))
        doc.add_node(NodeModel(id="act", type_name="relu"))
        doc.add_edge(EdgeModel(id="e1", source_node="in", source_port="value", target_node="lin", target_port="x"))
        doc.add_edge(EdgeModel(id="e2", source_node="lin", source_port="output", target_node="act", target_port="x"))
        out = execute(doc)
        # identity linear then relu([1,-2,3]) = [1,0,3]
        self.assertEqual(out["act"]["output"].to_list(), [1.0, 0.0, 3.0])

    def test_neural_registered(self):
        self.assertIn("linear", default_registry())
        self.assertIn("relu", default_registry())


if __name__ == "__main__":
    unittest.main()
