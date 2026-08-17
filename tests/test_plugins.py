"""Plugin system tests: loading + node registration from plugins/. """
import unittest

import entropia_riko.nodes  # noqa: F401  触发内置节点注册
from entropia_riko.core.tensor import TensorValue
from entropia_riko.plugins.loader import load_plugins, loaded_plugins
from entropia_riko.runtime.registry import default_registry


class TestPlugins(unittest.TestCase):
    def test_load_plugins(self):
        plugins = load_plugins()
        self.assertIs(plugins, loaded_plugins)
        names = [p["name"] for p in plugins]
        self.assertIn("example_plugin", names)
        example = next(p for p in plugins if p["name"] == "example_plugin")
        self.assertEqual(example["status"], "loaded")

    def test_plugin_node_registered(self):
        load_plugins()
        self.assertIn("plugin_double", default_registry())

    def test_plugin_node_executes(self):
        load_plugins()
        node = default_registry().get("plugin_double")()
        out = node.execute({"x": TensorValue([1.0, 2.0, 3.0])}, node.params, {})
        self.assertEqual(out["result"].to_list(), [2.0, 4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
