"""TensorFlow / Keras nodes (optional backend).

Keras layers and TensorFlow ops. ``import tensorflow`` is deferred to
``execute()`` so the editor still registers these nodes (and runs its torch
backend) when TensorFlow is not installed — executing a TF node then raises a
clear error instead.

Convention: Keras convolution layers use Keras' native **channels-last**
layout ``(batch, height, width, channels)`` (unlike torch's channels-first).
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import BaseNode, NodeInput, NodeOutput, Parameter
from ...runtime.registry import register
from ...backend.tf_converter import to_tf, from_tf


def _tf():
    import tensorflow as tf
    return tf


# ------------------------------------------------------------------ factories
def _tf_unary(_t, _l, _fn, _cat="Neural"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _cat
        inputs = [NodeInput("x", required=True)]
        outputs = [NodeOutput("result")]
        parameters = []

        def execute(self, inputs, params, context):
            tf = _tf()
            x = to_tf(inputs["x"])
            return {"result": from_tf(_fn(tf, x), metadata={"backend": "tensorflow"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


def _tf_binary(_t, _l, _fn, _cat="Tensor"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _cat
        inputs = [NodeInput("left", required=True), NodeInput("right", required=True)]
        outputs = [NodeOutput("result")]
        parameters = []

        def execute(self, inputs, params, context):
            tf = _tf()
            a = to_tf(inputs["left"])
            b = to_tf(inputs["right"])
            return {"result": from_tf(_fn(tf, a, b), metadata={"backend": "tensorflow"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


# ------------------------------------------------------------------ tf ops
_tf_unary("keras_relu", "Keras ReLU", lambda tf, x: tf.nn.relu(x))
_tf_unary("keras_sigmoid", "Keras Sigmoid", lambda tf, x: tf.nn.sigmoid(x))
_tf_unary("keras_tanh", "Keras Tanh", lambda tf, x: tf.nn.tanh(x))
_tf_unary("keras_gelu", "Keras GELU", lambda tf, x: tf.nn.gelu(x))
_tf_binary("tf_add", "TF Add", lambda tf, a, b: tf.add(a, b))
_tf_binary("tf_multiply", "TF Multiply", lambda tf, a, b: tf.multiply(a, b))
_tf_binary("tf_matmul", "TF MatMul", lambda tf, a, b: tf.matmul(a, b))
_tf_binary("tf_concat", "TF Concat", lambda tf, a, b: tf.concat([a, b], axis=-1), "Shape")


@register("keras_softmax")
class KerasSoftmaxNode(BaseNode):
    type_name = "keras_softmax"
    label = "Keras Softmax"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("axis", default=-1, dtype="int")]

    def execute(self, inputs, params, context):
        tf = _tf()
        return {"result": from_tf(
            tf.nn.softmax(to_tf(inputs["x"]), axis=int(params["axis"])),
            metadata={"backend": "tensorflow"})}


@register("tf_reshape")
class TfReshapeNode(BaseNode):
    type_name = "tf_reshape"
    label = "TF Reshape"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("shape", kind="any", required=True)]

    def execute(self, inputs, params, context):
        tf = _tf()
        return {"result": from_tf(
            tf.reshape(to_tf(inputs["x"]), tuple(params["shape"])),
            metadata={"backend": "tensorflow"})}


@register("tf_transpose")
class TfTransposeNode(BaseNode):
    type_name = "tf_transpose"
    label = "TF Transpose"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("perm", kind="any", default=None)]

    def execute(self, inputs, params, context):
        tf = _tf()
        perm = tuple(params["perm"]) if params.get("perm") is not None else None
        return {"result": from_tf(
            tf.transpose(to_tf(inputs["x"]), perm=perm),
            metadata={"backend": "tensorflow"})}


@register("tf_reduce")
class TfReduceNode(BaseNode):
    type_name = "tf_reduce"
    label = "TF Reduce"
    category = "Reduce"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("op", kind="scalar", required=True, choices=["sum", "mean", "max", "min"]),
                  Parameter("axis", kind="any", default=None)]

    def execute(self, inputs, params, context):
        tf = _tf()
        x = to_tf(inputs["x"])
        op = params["op"]
        axis = tuple(params["axis"]) if isinstance(params.get("axis"), list) else params.get("axis")
        fn = {"sum": tf.reduce_sum, "mean": tf.reduce_mean,
              "max": tf.reduce_max, "min": tf.reduce_min}[op]
        y = fn(x, axis=axis) if axis is not None else fn(x)
        return {"result": from_tf(y, metadata={"backend": "tensorflow"})}


# ------------------------------------------------------------------ Keras layers
@register("keras_dense")
class KerasDenseNode(BaseNode):
    type_name = "keras_dense"
    label = "Keras Dense"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [Parameter("units", required=True, dtype="int"),
                  Parameter("activation", default=None)]

    def execute(self, inputs, params, context):
        tf = _tf()
        layer = tf.keras.layers.Dense(int(params["units"]), activation=params.get("activation") or None)
        return {"output": from_tf(layer(to_tf(inputs["x"])), metadata={"backend": "tensorflow"})}


@register("keras_conv2d")
class KerasConv2dNode(BaseNode):
    type_name = "keras_conv2d"
    label = "Keras Conv2D"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]  # channels-last (B,H,W,C)
    outputs = [NodeOutput("output")]
    parameters = [Parameter("filters", required=True, dtype="int"),
                  Parameter("kernel_size", required=True, dtype="int"),
                  Parameter("strides", default=1, dtype="int"),
                  Parameter("padding", default="same")]

    def execute(self, inputs, params, context):
        tf = _tf()
        layer = tf.keras.layers.Conv2D(int(params["filters"]), int(params["kernel_size"]),
                                       strides=int(params["strides"]), padding=params.get("padding", "same"))
        return {"output": from_tf(layer(to_tf(inputs["x"])), metadata={"backend": "tensorflow"})}


@register("keras_flatten")
class KerasFlattenNode(BaseNode):
    type_name = "keras_flatten"
    label = "Keras Flatten"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("output")]
    parameters = []

    def execute(self, inputs, params, context):
        tf = _tf()
        return {"output": from_tf(tf.keras.layers.Flatten()(to_tf(inputs["x"])),
                                  metadata={"backend": "tensorflow"})}


@register("keras_maxpool2d")
class KerasMaxPool2dNode(BaseNode):
    type_name = "keras_maxpool2d"
    label = "Keras MaxPool2D"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [Parameter("pool_size", required=True, dtype="int"),
                  Parameter("strides", kind="any", default=None)]

    def execute(self, inputs, params, context):
        tf = _tf()
        ps = int(params["pool_size"])
        strides = int(params["strides"]) if params.get("strides") is not None else ps
        layer = tf.keras.layers.MaxPooling2D(pool_size=ps, strides=strides)
        return {"output": from_tf(layer(to_tf(inputs["x"])), metadata={"backend": "tensorflow"})}


@register("keras_avgpool2d")
class KerasAvgPool2dNode(BaseNode):
    type_name = "keras_avgpool2d"
    label = "Keras AvgPool2D"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [Parameter("pool_size", required=True, dtype="int"),
                  Parameter("strides", kind="any", default=None)]

    def execute(self, inputs, params, context):
        tf = _tf()
        ps = int(params["pool_size"])
        strides = int(params["strides"]) if params.get("strides") is not None else ps
        layer = tf.keras.layers.AveragePooling2D(pool_size=ps, strides=strides)
        return {"output": from_tf(layer(to_tf(inputs["x"])), metadata={"backend": "tensorflow"})}


@register("keras_embedding")
class KerasEmbeddingNode(BaseNode):
    type_name = "keras_embedding"
    label = "Keras Embedding"
    category = "Neural"
    inputs = [NodeInput("indices", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [Parameter("input_dim", required=True, dtype="int"),
                  Parameter("output_dim", required=True, dtype="int")]

    def execute(self, inputs, params, context):
        tf = _tf()
        layer = tf.keras.layers.Embedding(int(params["input_dim"]), int(params["output_dim"]))
        return {"output": from_tf(layer(to_tf(inputs["indices"])), metadata={"backend": "tensorflow"})}


@register("keras_layernorm")
class KerasLayerNormNode(BaseNode):
    type_name = "keras_layernorm"
    label = "Keras LayerNorm"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [Parameter("axis", default=-1, dtype="int")]

    def execute(self, inputs, params, context):
        tf = _tf()
        layer = tf.keras.layers.LayerNormalization(axis=int(params["axis"]))
        return {"output": from_tf(layer(to_tf(inputs["x"])), metadata={"backend": "tensorflow"})}


@register("keras_dropout")
class KerasDropoutNode(BaseNode):
    type_name = "keras_dropout"
    label = "Keras Dropout"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [Parameter("rate", default=0.5, dtype="float")]

    def execute(self, inputs, params, context):
        tf = _tf()
        layer = tf.keras.layers.Dropout(float(params["rate"]))
        return {"output": from_tf(layer(to_tf(inputs["x"]), training=False),
                                  metadata={"backend": "tensorflow"})}
