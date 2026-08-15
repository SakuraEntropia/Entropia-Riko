"""Default plugin: extra elementwise math nodes.

Follows the same public node API as ``plugins/example_plugin/nodes.py``:
import ``src.*`` modules and register node classes with ``@register``.
"""
from src.runtime.registry import register
from src.nodes.base import BaseNode, NodeInput, NodeOutput, Parameter
from src.core.tensor import TensorValue, broadcast_op


@register("plugin_square")
class PluginSquareNode(BaseNode):
    type_name = "plugin_square"
    label = "Square (plugin)"
    category = "Plugin"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = []

    def execute(self, inputs, params, context):
        x = inputs["x"]
        data = broadcast_op(x.data, x.shape, x.data, x.shape, lambda a, b: a * b)
        return {"result": TensorValue(data, shape=x.shape, dtype=x.dtype)}


@register("plugin_scale")
class PluginScaleNode(BaseNode):
    type_name = "plugin_scale"
    label = "Scale (plugin)"
    category = "Plugin"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = [
        Parameter("factor", kind="scalar", default=2.0, required=False, dtype="float")
    ]

    def execute(self, inputs, params, context):
        x = inputs["x"]
        factor = float(params.get("factor", 2.0))
        data = broadcast_op(x.data, x.shape, factor, (), lambda a, b: a * b)
        return {"result": TensorValue(data, shape=x.shape, dtype=x.dtype)}
