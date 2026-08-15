"""Subgraph nodes: graph_input, graph_output, graph_reference, import.

- ``graph_input`` / ``graph_output`` define a subgraph's tensor interface.
- ``graph_reference`` references another ``.riko`` file by path.
- ``import`` references another ``.riko`` file by module name (Python-style
  ``import xx``), resolved against the module search path.

Both reference nodes execute the referenced graph with a single tensor input
(``graph_input`` name='input') and return a single output (``graph_output``
name='output').
"""
from __future__ import annotations

from typing import Any, Dict

from ..base import BaseNode, NodeInput, NodeOutput, Parameter
from ...runtime.registry import register
from ...runtime.subgraph import run_subgraph


@register("graph_input")
class GraphInputNode(BaseNode):
    """Defines an input port for a subgraph. Receives value from context.

    ``data_kind`` declares the multimodal kind of this input (tensor / text /
    json / image_tensor); the value itself is passed through unchanged.
    """

    type_name = "graph_input"
    label = "Graph Input"
    category = "Subgraph"
    inputs = []
    outputs = [NodeOutput("value", data_kind="tensor")]
    parameters = [
        Parameter("name", kind="scalar", required=True),
        Parameter("data_kind", kind="scalar", default="tensor",
                  choices=["tensor", "text", "json", "image_tensor"]),
    ]

    def execute(self, inputs, params, context):
        name = params["name"]
        graph_inputs = context.get("graph_inputs", {})
        if name not in graph_inputs:
            raise ValueError(
                f"what: 图输入 '{name}' 未提供。\n"
                f"where: nodes.subgraph.graph_input.execute\n"
                f"how_to_fix: 调用方需在 context.graph_inputs 中提供 '{name}'。"
            )
        return {"value": graph_inputs[name]}


@register("graph_output")
class GraphOutputNode(BaseNode):
    """Defines an output port for a subgraph. Stores value to context.

    ``data_kind`` declares the multimodal kind of this output; the value is
    passed through unchanged.
    """

    type_name = "graph_output"
    label = "Graph Output"
    category = "Subgraph"
    inputs = [NodeInput("value", data_kind="tensor", required=True)]
    outputs = []
    parameters = [
        Parameter("name", kind="scalar", required=True),
        Parameter("data_kind", kind="scalar", default="tensor",
                  choices=["tensor", "text", "json", "image_tensor"]),
    ]

    def execute(self, inputs, params, context):
        name = params["name"]
        context.setdefault("graph_outputs", {})[name] = inputs["value"]
        return {}


@register("graph_reference")
class GraphReferenceNode(BaseNode):
    """References another ``.riko`` graph file by path and executes it.

    Multi-port: the ``input`` / ``input_2`` / ``input_3`` ports feed the
    referenced graph's ``graph_input`` nodes of the same name; the ``output`` /
    ``output_2`` / ``output_3`` ports carry the referenced graph's
    ``graph_output`` values of the same name.
    """

    type_name = "graph_reference"
    label = "Graph Reference"
    category = "Subgraph"
    inputs = [
        NodeInput("input", data_kind="tensor", required=False),
        NodeInput("input_2", data_kind="tensor", required=False),
        NodeInput("input_3", data_kind="tensor", required=False),
    ]
    outputs = [
        NodeOutput("output", data_kind="tensor"),
        NodeOutput("output_2", data_kind="tensor"),
        NodeOutput("output_3", data_kind="tensor"),
    ]
    parameters = [
        Parameter("file", kind="scalar", required=True),
        Parameter("device", kind="scalar", default="cpu"),
    ]

    def execute(self, inputs, params, context):
        return run_subgraph(params["file"], inputs, context)


@register("import")
class ImportNode(BaseNode):
    """Import another ``.riko`` graph by module name (Python-style ``import``).

    The ``module`` parameter is a bare name (e.g. ``mlp``) resolved against
    ``MODULE_SEARCH_PATHS`` (workflows/, examples/, examples/models/), or a
    relative path to a ``.riko`` file. Multi-port like ``graph_reference``.
    """

    type_name = "import"
    label = "Import"
    category = "Subgraph"
    inputs = [
        NodeInput("input", data_kind="tensor", required=False),
        NodeInput("input_2", data_kind="tensor", required=False),
        NodeInput("input_3", data_kind="tensor", required=False),
    ]
    outputs = [
        NodeOutput("output", data_kind="tensor"),
        NodeOutput("output_2", data_kind="tensor"),
        NodeOutput("output_3", data_kind="tensor"),
    ]
    parameters = [Parameter("module", kind="scalar", required=True)]

    def execute(self, inputs, params, context):
        return run_subgraph(params["module"], inputs, context)
