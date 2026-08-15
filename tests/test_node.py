"""Stage 1 tests: tensor IR, document model, nodes.

运行：python -m unittest discover -s tests -t .
"""
import unittest

import src.nodes  # noqa: F401  触发 constant/add/multiply 注册
from src.core.tensor import (
    TensorValue,
    infer_shape,
    broadcast_shapes,
    broadcast_op,
)
from src.core.document import GraphDocument, NodeModel, EdgeModel, PortModel
from src.nodes.math.constant import ConstantNode
from src.nodes.math.add import AddNode
from src.nodes.math.multiply import MultiplyNode


class TestTensor(unittest.TestCase):
    def test_infer_shape(self):
        self.assertEqual(infer_shape(3.0), ())
        self.assertEqual(infer_shape([1, 2, 3]), (3,))
        self.assertEqual(infer_shape([[1, 2], [3, 4]]), (2, 2))

    def test_scalar(self):
        t = TensorValue(5.0)
        self.assertEqual(t.shape, ())
        self.assertEqual(t.data_kind, "scalar")
        self.assertEqual(t.item(), 5.0)
        self.assertIn("5.0", t.summary())

    def test_from_value(self):
        t = TensorValue.from_value([1, 2, 3], dtype="int32")
        self.assertEqual(t.shape, (3,))
        self.assertEqual(t.to_list(), [1, 2, 3])

    def test_broadcast(self):
        self.assertEqual(
            broadcast_op([1, 2, 3], (3,), 10, (), lambda a, b: a + b),
            [11, 12, 13],
        )

    def test_broadcast_shapes(self):
        self.assertEqual(broadcast_shapes((2, 3), (3,)), (2, 3))

    def test_eq(self):
        self.assertEqual(TensorValue(2.0), TensorValue(2.0))
        self.assertNotEqual(TensorValue(2.0), TensorValue(3.0))


class TestDocument(unittest.TestCase):
    def test_node_roundtrip(self):
        n = NodeModel(
            id="a",
            type_name="constant",
            label="Constant",
            category="Inputs",
            position=(10, 20),
            parameters={"value": 1.0},
            inputs=[],
            outputs=[PortModel("value", "Value", "tensor", "out")],
        )
        n2 = NodeModel.from_dict(n.to_dict())
        self.assertEqual(n2.id, "a")
        self.assertEqual(n2.position, (10, 20))
        self.assertEqual(n2.parameters, {"value": 1.0})
        self.assertEqual(len(n2.outputs), 1)
        self.assertEqual(n2.outputs[0].name, "value")

    def test_graph_doc_roundtrip(self):
        doc = GraphDocument()
        doc.add_node(NodeModel(id="a", type_name="constant"))
        doc.add_node(NodeModel(id="b", type_name="add"))
        doc.add_edge(EdgeModel(
            id="e1", source_node="a", source_port="value",
            target_node="b", target_port="left",
        ))
        doc2 = GraphDocument.from_dict(doc.to_dict())
        self.assertEqual(len(doc2.nodes), 2)
        self.assertEqual(len(doc2.edges), 1)

    def test_remove_node_cascades_edges(self):
        doc = GraphDocument()
        doc.add_node(NodeModel(id="a", type_name="constant"))
        doc.add_node(NodeModel(id="b", type_name="add"))
        doc.add_edge(EdgeModel(
            id="e1", source_node="a", source_port="value",
            target_node="b", target_port="left",
        ))
        doc.remove_node("a")
        self.assertNotIn("a", doc.nodes)
        self.assertEqual(len(doc.edges), 0)

    def test_duplicate_node_id(self):
        doc = GraphDocument()
        doc.add_node(NodeModel(id="a", type_name="constant"))
        with self.assertRaises(ValueError):
            doc.add_node(NodeModel(id="a", type_name="constant"))

    def test_binary_roundtrip(self):
        """The binary .ric format round-trips losslessly."""
        doc = GraphDocument()
        doc.add_node(NodeModel(id="a", type_name="constant", parameters={"value": 2.0}))
        doc.add_node(NodeModel(id="b", type_name="add"))
        doc.add_edge(EdgeModel(id="e1", source_node="a", source_port="value",
                               target_node="b", target_port="left"))
        raw = doc.to_binary()
        self.assertIsInstance(raw, bytes)
        self.assertTrue(raw.startswith(b"ERIK"))
        self.assertEqual(GraphDocument.from_binary(raw).to_dict(), doc.to_dict())

    def test_binary_rejects_garbage(self):
        with self.assertRaises(ValueError):
            GraphDocument.from_binary(b"not-a-rik")


class TestNodes(unittest.TestCase):
    def test_constant(self):
        node = ConstantNode({"value": 42.0})
        out = node.execute({}, node.params, {})
        self.assertEqual(out["value"], TensorValue(42.0))

    def test_constant_requires_value(self):
        with self.assertRaises(ValueError):
            ConstantNode({})

    def test_add(self):
        node = AddNode()
        out = node.execute(
            {"left": TensorValue(2.0), "right": TensorValue(3.0)},
            node.params, {},
        )
        self.assertEqual(out["result"].item(), 5.0)

    def test_add_broadcast(self):
        node = AddNode()
        out = node.execute(
            {"left": TensorValue([1, 2, 3]), "right": TensorValue(10)},
            node.params, {},
        )
        self.assertEqual(out["result"].to_list(), [11, 12, 13])

    def test_multiply(self):
        node = MultiplyNode()
        out = node.execute(
            {"left": TensorValue(4.0), "right": TensorValue(5.0)},
            node.params, {},
        )
        self.assertEqual(out["result"].item(), 20.0)

    def test_node_labels(self):
        self.assertEqual(ConstantNode.label, "Constant")
        self.assertEqual(AddNode.label, "Add")
        self.assertEqual(MultiplyNode.label, "Multiply")
        self.assertEqual(AddNode.category, "Math")


if __name__ == "__main__":
    unittest.main()
