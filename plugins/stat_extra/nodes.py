"""Default plugin: affine transform + negation nodes."""
from src.runtime.registry import register
from src.nodes.base import BaseNode, NodeInput, NodeOutput, Parameter
from src.core.tensor import TensorValue, broadcast_op


@register("plugin_shift_scale")
class PluginShiftScaleNode(BaseNode):
    type_name = "plugin_shift_scale"
    label = "Shift + Scale (plugin)"
    category = "Plugin"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = [
        Parameter("scale", kind="scalar", default=1.0, required=False, dtype="float"),
        Parameter("shift", kind="scalar", default=0.0, required=False, dtype="float"),
    ]

    def execute(self, inputs, params, context):
        x = inputs["x"]
        scale = float(params.get("scale", 1.0))
        shift = float(params.get("shift", 0.0))
        data = broadcast_op(x.data, x.shape, scale, (), lambda a, b: a * b)
        data = broadcast_op(data, x.shape, shift, (), lambda a, b: a + b)
        return {"result": TensorValue(data, shape=x.shape, dtype=x.dtype)}


@register("plugin_neg")
class PluginNegNode(BaseNode):
    type_name = "plugin_neg"
    label = "Negate (plugin)"
    category = "Plugin"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = []

    def execute(self, inputs, params, context):
        x = inputs["x"]
        data = broadcast_op(x.data, x.shape, -1, (), lambda a, b: a * b)
        return {"result": TensorValue(data, shape=x.shape, dtype=x.dtype)}
