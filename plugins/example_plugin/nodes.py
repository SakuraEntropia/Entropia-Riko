"""Example plugin node.

Imports the public node API via absolute ``src.*`` paths (the server runs from
the project root, so ``src`` is importable) and registers a node through the
``@register`` decorator.
"""
from entropia_riko.runtime.registry import register
from entropia_riko.nodes.base import BaseNode, NodeInput, NodeOutput
from entropia_riko.core.tensor import TensorValue, broadcast_op, broadcast_shapes


@register("plugin_double")
class PluginDoubleNode(BaseNode):
    type_name = "plugin_double"
    label = "Double (plugin)"
    category = "Plugin"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = []

    def execute(self, inputs, params, context):
        x = inputs["x"]
        data = broadcast_op(x.data, x.shape, 2, (), lambda a, b: a * b)
        return {"result": TensorValue(data, shape=x.shape, dtype=x.dtype)}
