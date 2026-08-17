"""Stage 1 tests: registry and runtime executor.

运行：python -m unittest discover -s tests -t .
"""
import unittest

import entropia_riko.nodes  # noqa: F401  触发节点注册
from entropia_riko.core.document import EdgeModel, GraphDocument, NodeModel
from entropia_riko.nodes.base import BaseNode
from entropia_riko.runtime.executor import (
    RuntimeExecutionError,
    execute,
    execution_order,
    validate,
)
from entropia_riko.runtime.registry import Registry, default_registry


class TestRegistry(unittest.TestCase):
    def test_default_has_builtins(self):
        reg = default_registry()
        self.assertIn("constant", reg)
        self.assertIn("add", reg)
        self.assertIn("multiply", reg)

    def test_list_sorted(self):
        reg = default_registry()
        names = reg.list()
        self.assertEqual(names, sorted(names))

    def test_duplicate_rejected(self):
        reg = Registry()

        class A(BaseNode):
            type_name = "a"

        reg.register("x", A)
        with self.assertRaises(ValueError):
            reg.register("x", A)

    def test_unknown_lookup(self):
        reg = Registry()
        with self.assertRaises(KeyError):
            reg.get("nope")


class TestExecutor(unittest.TestCase):
    def _build_chain(self):
        # constant(2) -> add.left ; constant(3) -> add.right
        # add.result  -> multiply.left ; constant(4) -> multiply.right
        # (2 + 3) * 4 = 20
        doc = GraphDocument()
        doc.add_node(NodeModel(id="a", type_name="constant", parameters={"value": 2.0}))
        doc.add_node(NodeModel(id="b", type_name="constant", parameters={"value": 3.0}))
        doc.add_node(NodeModel(id="sum", type_name="add"))
        doc.add_node(NodeModel(id="c", type_name="constant", parameters={"value": 4.0}))
        doc.add_node(NodeModel(id="prod", type_name="multiply"))
        doc.add_edge(EdgeModel(id="e1", source_node="a", source_port="value", target_node="sum", target_port="left"))
        doc.add_edge(EdgeModel(id="e2", source_node="b", source_port="value", target_node="sum", target_port="right"))
        doc.add_edge(EdgeModel(id="e3", source_node="sum", source_port="result", target_node="prod", target_port="left"))
        doc.add_edge(EdgeModel(id="e4", source_node="c", source_port="value", target_node="prod", target_port="right"))
        return doc

    def test_execute_chain(self):
        out = execute(self._build_chain())
        self.assertEqual(out["sum"]["result"].item(), 5.0)
        self.assertEqual(out["prod"]["result"].item(), 20.0)

    def test_execution_order(self):
        order = execution_order(self._build_chain())
        self.assertEqual(order[-1], "prod")
        self.assertEqual(len(order), 5)

    def test_validate_missing_input(self):
        doc = GraphDocument()
        doc.add_node(NodeModel(id="sum", type_name="add"))
        errors = validate(doc)
        self.assertTrue(any("缺少必需输入" in e for e in errors))

    def test_validate_unknown_type(self):
        doc = GraphDocument()
        doc.add_node(NodeModel(id="x", type_name="nope"))
        errors = validate(doc)
        self.assertTrue(any("未注册" in e for e in errors))

    def test_validate_cycle(self):
        doc = GraphDocument()
        doc.add_node(NodeModel(id="x", type_name="add"))
        doc.add_node(NodeModel(id="y", type_name="add"))
        doc.add_edge(EdgeModel(id="e1", source_node="x", source_port="result", target_node="y", target_port="left"))
        doc.add_edge(EdgeModel(id="e2", source_node="x", source_port="result", target_node="y", target_port="right"))
        doc.add_edge(EdgeModel(id="e3", source_node="y", source_port="result", target_node="x", target_port="left"))
        doc.add_edge(EdgeModel(id="e4", source_node="y", source_port="result", target_node="x", target_port="right"))
        errors = validate(doc)
        self.assertTrue(any("环" in e for e in errors))

    def test_execute_invalid_raises(self):
        doc = GraphDocument()
        doc.add_node(NodeModel(id="sum", type_name="add"))
        with self.assertRaises(RuntimeExecutionError):
            execute(doc)

    def test_validate_bad_port(self):
        doc = GraphDocument()
        doc.add_node(NodeModel(id="a", type_name="constant", parameters={"value": 1.0}))
        doc.add_node(NodeModel(id="b", type_name="add"))
        doc.add_edge(EdgeModel(id="e1", source_node="a", source_port="nope", target_node="b", target_port="left"))
        errors = validate(doc)
        self.assertTrue(any("无输出端口" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
