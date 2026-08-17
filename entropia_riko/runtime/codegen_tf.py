"""Export a graph to a ``tf.keras.Model`` script.

Handles both the Keras-native node set (``keras_*`` / ``tf_*``) **and** the
standard PyTorch node set, mapping torch nodes to their TensorFlow/Keras
equivalents so a torch-style graph still exports to clean, runnable Keras code
(instead of ``None`` placeholders).

Notes:

- Learnable layers go in ``__init__``; inline ops run in ``call``.
- Convolution / pooling follow Keras' **channels-last** ``(B, H, W, C)``
  convention. If a torch graph is channels-first ``(B, C, H, W)``, insert a
  ``tf_transpose`` node to convert layouts.
- TensorFlow is not required to *generate* the script (pure text generation).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.document import GraphDocument
from .codegen import _dim, _lit, _sanitize, _tuple_lit
from .executor import execution_order
from .registry import Registry, default_registry


def _keras_pad(pad: Any) -> str:
    """torch int padding → Keras padding ("valid" for 0, else the int)."""
    p = int(pad or 0)
    return '"valid"' if p == 0 else str(p)


# ------------------------------------------------------------------ layers
def _keras_ctor(t: str, p: Dict[str, Any]) -> str:
    # --- Keras-native ---
    if t == "keras_dense":
        act = f", activation={_lit(p['activation'])}" if p.get("activation") else ""
        return f"tf.keras.layers.Dense({p['units']}{act})"
    if t == "keras_conv2d":
        return (f"tf.keras.layers.Conv2D({p['filters']}, {p['kernel_size']}, "
                f"strides={p.get('strides', 1)}, padding={_lit(p.get('padding', 'same'))})")
    if t == "keras_flatten":
        return "tf.keras.layers.Flatten()"
    if t == "keras_maxpool2d":
        ps = p["pool_size"]
        st = p.get("strides") or ps
        return f"tf.keras.layers.MaxPooling2D(pool_size={ps}, strides={st})"
    if t == "keras_avgpool2d":
        ps = p["pool_size"]
        st = p.get("strides") or ps
        return f"tf.keras.layers.AveragePooling2D(pool_size={ps}, strides={st})"
    if t == "keras_embedding":
        return f"tf.keras.layers.Embedding({p['input_dim']}, {p['output_dim']})"
    if t == "keras_layernorm":
        return f"tf.keras.layers.LayerNormalization(axis={p.get('axis', -1)})"
    if t == "keras_dropout":
        return f"tf.keras.layers.Dropout({p.get('rate', 0.5)})"

    # --- torch → Keras mappings ---
    if t == "linear":
        return f"tf.keras.layers.Dense({p.get('out_features')}, use_bias={p.get('use_bias', True)})"
    if t == "conv2d":
        return (f"tf.keras.layers.Conv2D({p.get('out_channels')}, {p.get('kernel_size')}, "
                f"strides={p.get('stride', 1)}, padding={_keras_pad(p.get('padding', 0))})")
    if t == "conv1d":
        return (f"tf.keras.layers.Conv1D({p.get('out_channels')}, {p.get('kernel_size')}, "
                f"strides={p.get('stride', 1)}, padding={_keras_pad(p.get('padding', 0))})")
    if t == "conv_transpose2d":
        return (f"tf.keras.layers.Conv2DTranspose({p.get('out_channels')}, {p.get('kernel_size')}, "
                f"strides={p.get('stride', 1)}, padding={_keras_pad(p.get('padding', 0))})")
    if t == "maxpool2d":
        ks = p.get("kernel_size")
        st = p.get("stride") or ks
        return f"tf.keras.layers.MaxPooling2D(pool_size={ks}, strides={st})"
    if t == "avgpool2d":
        ks = p.get("kernel_size")
        st = p.get("stride") or ks
        return f"tf.keras.layers.AveragePooling2D(pool_size={ks}, strides={st})"
    if t == "embedding":
        return f"tf.keras.layers.Embedding({p.get('num_embeddings')}, {p.get('embedding_dim')})"
    if t == "dropout":
        return f"tf.keras.layers.Dropout({p.get('p', 0.5)})"
    if t in ("batchnorm1d", "batchnorm2d"):
        return "tf.keras.layers.BatchNormalization(axis=1)"
    if t == "layernorm":
        return "tf.keras.layers.LayerNormalization()"
    if t == "groupnorm":
        return (f"tf.keras.layers.GroupNormalization(groups={p.get('num_groups')}, "
                f"channels={p.get('num_channels')})")
    if t == "multihead_attention":
        heads = int(p.get("num_heads", 1))
        emb = int(p.get("embed_dim", heads))
        return f"tf.keras.layers.MultiHeadAttention(num_heads={heads}, key_dim={emb // heads})"
    if t == "transformer_encoder":
        raise ValueError("transformer_encoder 暂不支持 TF 导出")
    raise ValueError(f"未知层类型 {t}")


# torch node types that become Keras layers (declared in __init__).
_TORCH_NN = {
    "linear", "conv2d", "conv1d", "conv_transpose2d", "maxpool2d", "avgpool2d",
    "embedding", "dropout", "batchnorm1d", "batchnorm2d", "layernorm", "groupnorm",
    "multihead_attention",
}
_KERAS_LAYERS = {
    "keras_dense", "keras_conv2d", "keras_flatten", "keras_maxpool2d",
    "keras_avgpool2d", "keras_embedding", "keras_layernorm", "keras_dropout",
}

# Inline TF call: type → expression template (uses {x}/{a}/{b}).
_TF_UNARY = {
    "abs": "tf.abs", "exp": "tf.exp", "log": "tf.math.log", "sqrt": "tf.sqrt",
    "sign": "tf.sign", "cos": "tf.cos", "sin": "tf.sin",
    "reciprocal": "tf.math.reciprocal", "floor": "tf.floor", "ceil": "tf.math.ceil",
    "round": "tf.round", "erf": "tf.math.erf", "erfc": "tf.math.erfc",
    "log1p": "tf.math.log1p", "expm1": "tf.math.expm1",
    "relu": "tf.nn.relu", "sigmoid": "tf.nn.sigmoid", "tanh": "tf.nn.tanh",
    "gelu": "tf.nn.gelu", "silu": "tf.nn.silu", "elu": "tf.nn.elu", "selu": "tf.nn.selu",
    "softplus": "tf.nn.softplus", "keras_relu": "tf.nn.relu",
    "keras_sigmoid": "tf.nn.sigmoid", "keras_tanh": "tf.nn.tanh", "keras_gelu": "tf.nn.gelu",
}

_TF_BINARY = {
    "add": "tf.add", "multiply": "tf.multiply", "sub": "tf.subtract", "div": "tf.divide",
    "matmul": "tf.matmul", "mm": "tf.linalg.matmul", "maximum": "tf.maximum",
    "minimum": "tf.minimum", "pow": "tf.pow", "torch_add": "tf.add", "torch_multiply": "tf.multiply",
    "tf_add": "tf.add", "tf_multiply": "tf.multiply", "tf_matmul": "tf.matmul",
}


def export_keras(doc: GraphDocument, registry: Optional[Registry] = None) -> str:
    """Generate a clean, self-contained tf.keras.Model Python script."""
    reg = registry if registry is not None else default_registry()
    try:
        order = execution_order(doc)
    except Exception as exc:
        return f"# ERROR: {type(exc).__name__}: {exc}\n"

    # 1. call() parameters from graph_input nodes.
    params: List[str] = []
    seen: Set[str] = set()
    for nid in order:
        node = doc.nodes[nid]
        if node.type_name != "graph_input":
            continue
        base = _sanitize(node.parameters.get("name", "x"))
        name = base
        i = 1
        while name in seen:
            i += 1
            name = f"{base}_{i}"
        seen.add(name)
        params.append(name)
    if not params:
        params = ["inputs"]

    # 2. assign variable names.
    used: Set[str] = set(params)
    out_vars: Dict[Tuple[str, str], str] = {}
    node_vars: Dict[str, Dict[str, str]] = {}

    def _unique(base: str) -> str:
        name = base
        i = 1
        while name in used:
            i += 1
            name = f"{base}_{i}"
        used.add(name)
        return name

    gi = 0
    for nid in order:
        node = doc.nodes[nid]
        t = node.type_name
        if t == "graph_input":
            out_vars[(nid, "value")] = params[gi]
            node_vars[nid] = {"value": params[gi]}
            gi += 1
            continue
        cls = reg.get(t) if t in reg else None
        ports = [o.name for o in cls.outputs] if cls else ["result"]
        if t == "graph_output" or not ports:
            continue
        base = _unique(_sanitize(nid))
        if len(ports) == 1:
            out_vars[(nid, ports[0])] = base
            node_vars[nid] = {ports[0]: base}
        else:
            mapping = {port: f"{base}_{port}" for port in ports}
            node_vars[nid] = mapping
            for port, var in mapping.items():
                out_vars[(nid, port)] = var

    # 3. incoming edges.
    in_edges: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for e in doc.edges:
        in_edges[(e.target_node, e.target_port)] = (e.source_node, e.source_port)

    def _source_var(node: str, port: str) -> str:
        ports = node_vars.get(node)
        if not ports:
            return "_"
        if port in ports:
            return ports[port]
        if len(ports) == 1:
            return next(iter(ports.values()))
        return "_"

    init_lines: List[str] = []
    call_lines: List[str] = []
    returns: List[str] = []
    last_var: Optional[str] = None

    for nid in order:
        node = doc.nodes[nid]
        t = node.type_name
        p = dict(node.parameters)
        cls = reg.get(t) if t in reg else None

        iv: Dict[str, str] = {}
        if cls:
            for inp in cls.inputs:
                key = (nid, inp.name)
                if key in in_edges:
                    iv[inp.name] = _source_var(*in_edges[key])
                elif inp.default is not None:
                    iv[inp.name] = _lit(inp.default)

        ports = [o.name for o in cls.outputs] if cls else ["result"]
        dst = out_vars.get((nid, ports[0])) if ports else None

        if t == "graph_input":
            continue
        if t == "graph_output":
            returns.append(iv.get("value", "None"))
            continue
        if dst is not None:
            last_var = dst

        # ---------------- constants ----------------
        if t == "constant":
            call_lines.append(f"{dst} = tf.constant({_lit(p.get('value'))}, dtype=tf.float32)")

        # ---------------- learnable layers (__init__) ----------------
        elif t in _KERAS_LAYERS or t in _TORCH_NN:
            try:
                init_lines.append(f"self.{dst} = {_keras_ctor(t, p)}")
            except ValueError as exc:
                call_lines.append(f"{dst} = None  # {exc}")
                continue
            inp = "indices" if t in ("embedding", "keras_embedding") else "x"
            x = iv.get(inp, "_")
            if t == "multihead_attention":
                call_lines.append(f"{dst} = self.{dst}({x}, {x})[0]")
            else:
                call_lines.append(f"{dst} = self.{dst}({x})")

        # ---------------- unary / activations ----------------
        elif t in _TF_UNARY:
            call_lines.append(f"{dst} = {_TF_UNARY[t]}({iv.get('x', '_')})")
        elif t == "neg":
            call_lines.append(f"{dst} = -{iv.get('x', '_')}")
        elif t == "square":
            call_lines.append(f"{dst} = {iv.get('x', '_')} * {iv.get('x', '_')}")
        elif t == "log2":
            call_lines.append(f"{dst} = tf.math.log({iv.get('x', '_')}) / tf.math.log(2.0)")
        elif t == "log10":
            call_lines.append(f"{dst} = tf.math.log({iv.get('x', '_')}) / tf.math.log(10.0)")
        elif t in ("softmax", "log_softmax", "keras_softmax"):
            fn = "tf.nn.softmax" if "softmax" in t and "log" not in t else "tf.nn.log_softmax"
            call_lines.append(f"{dst} = {fn}({iv.get('x', '_')}, axis={p.get('dim', p.get('axis', -1))})")
        elif t == "leaky_relu":
            call_lines.append(f"{dst} = tf.nn.leaky_relu({iv.get('x', '_')}, alpha={p.get('negative_slope', 0.01)})")
        elif t == "mish":
            call_lines.append(f"{dst} = {iv.get('x', '_')} * tf.math.tanh(tf.nn.softplus({iv.get('x', '_')}))")
        elif t == "glu":
            call_lines.append(f"{dst} = {iv.get('x', '_')} * tf.nn.sigmoid({iv.get('x', '_')})")
        elif t in ("hardswish", "hardsigmoid"):
            x = iv.get("x", "_")
            if t == "hardswish":
                call_lines.append(f"{dst} = {x} * tf.nn.relu6({x} + 3.0) / 6.0")
            else:
                call_lines.append(f"{dst} = tf.clip_by_value({x} / 6.0 + 0.5, 0.0, 1.0)")

        # ---------------- binary ----------------
        elif t in _TF_BINARY:
            a, b = iv.get("left", "_"), iv.get("right", "_")
            call_lines.append(f"{dst} = {_TF_BINARY[t]}({a}, {b})")
        elif t == "tf_concat":
            call_lines.append(f"{dst} = tf.concat([{iv.get('left', '_')}, {iv.get('right', '_')}], axis=-1)")
        elif t == "cross":
            call_lines.append(f"{dst} = tf.linalg.cross({iv.get('left', '_')}, {iv.get('right', '_')})")

        # ---------------- reduce ----------------
        elif t in ("sum", "mean", "max_reduce", "min_reduce", "prod", "argmax", "argmin"):
            fn = {"sum": "tf.reduce_sum", "mean": "tf.reduce_mean", "max_reduce": "tf.reduce_max",
                  "min_reduce": "tf.reduce_min", "prod": "tf.reduce_prod",
                  "argmax": "tf.argmax", "argmin": "tf.argmin"}[t]
            x = iv.get("x", "_")
            dim = p.get("dim")
            if dim is None:
                call_lines.append(f"{dst} = {fn}({x})")
            else:
                call_lines.append(f"{dst} = {fn}({x}, axis={_dim(dim)})")
        elif t == "tf_reduce":
            fn = {"sum": "tf.reduce_sum", "mean": "tf.reduce_mean",
                  "max": "tf.reduce_max", "min": "tf.reduce_min"}[p.get("op", "sum")]
            x = iv.get("x", "_")
            axis = p.get("axis")
            call_lines.append(f"{dst} = {fn}({x})" if axis is None else f"{dst} = {fn}({x}, axis={_dim(axis)})")

        # ---------------- shape ----------------
        elif t in ("reshape", "view"):
            call_lines.append(f"{dst} = tf.reshape({iv.get('x', '_')}, {_tuple_lit(p['shape'])})")
        elif t == "tf_reshape":
            call_lines.append(f"{dst} = tf.reshape({iv.get('x', '_')}, {_tuple_lit(p['shape'])})")
        elif t == "transpose":
            call_lines.append(f"{dst} = tf.transpose({iv.get('x', '_')}, perm=[{p.get('dim0')}, {p.get('dim1')}])")
        elif t == "permute":
            call_lines.append(f"{dst} = tf.transpose({iv.get('x', '_')}, perm={_tuple_lit(p['dims'])})")
        elif t == "tf_transpose":
            perm = _tuple_lit(p["perm"]) if p.get("perm") is not None else "None"
            call_lines.append(f"{dst} = tf.transpose({iv.get('x', '_')}, perm={perm})")
        elif t in ("flatten", "keras_flatten"):
            call_lines.append(f"{dst} = tf.keras.layers.Flatten()({iv.get('x', '_')})")
        elif t == "squeeze":
            x = iv.get("x", "_")
            call_lines.append(f"{dst} = tf.squeeze({x})" if p.get("dim") is None else f"{dst} = tf.squeeze({x}, axis={_dim(p['dim'])})")
        elif t == "unsqueeze":
            call_lines.append(f"{dst} = tf.expand_dims({iv.get('x', '_')}, axis={_dim(p['dim'])})")
        elif t == "concat":
            call_lines.append(f"{dst} = tf.concat([{iv.get('left', '_')}, {iv.get('right', '_')}], axis={_dim(p['dim'])})")
        elif t == "stack":
            call_lines.append(f"{dst} = tf.stack([{iv.get('left', '_')}, {iv.get('right', '_')}], axis={_dim(p['dim'])})")
        elif t in ("expand_as", "broadcast_to"):
            call_lines.append(f"{dst} = tf.broadcast_to({iv.get('x', '_')}, tf.shape({iv.get('other', iv.get('x', '_'))}))")

        # ---------------- creation ----------------
        elif t == "zeros":
            call_lines.append(f"{dst} = tf.zeros({_tuple_lit(p['shape'])})")
        elif t == "ones":
            call_lines.append(f"{dst} = tf.ones({_tuple_lit(p['shape'])})")
        elif t == "rand":
            call_lines.append(f"{dst} = tf.random.uniform({_tuple_lit(p['shape'])})")
        elif t == "randn":
            call_lines.append(f"{dst} = tf.random.normal({_tuple_lit(p['shape'])})")
        elif t == "eye":
            call_lines.append(f"{dst} = tf.eye({p.get('n')})")
        elif t == "arange":
            call_lines.append(f"{dst} = tf.range({p.get('start', 0)}, {p.get('end')}, {p.get('step', 1)})")
        elif t == "linspace":
            call_lines.append(f"{dst} = tf.linspace({p.get('start')}, {p.get('end')}, {p.get('steps')})")
        elif t == "full":
            call_lines.append(f"{dst} = tf.fill({_tuple_lit(p['shape'])}, {p.get('fill_value')})")
        elif t == "randint":
            call_lines.append(f"{dst} = tf.random.uniform({_tuple_lit(p['size'])}, minval={p.get('low', 0)}, maxval={p.get('high')}, dtype=tf.int32)")
        elif t == "randperm":
            call_lines.append(f"{dst} = tf.random.shuffle(tf.range({p.get('n')}))")

        # ---------------- losses ----------------
        elif t == "mse_loss":
            call_lines.append(f"{dst} = tf.reduce_mean(tf.square({iv.get('pred', '_')} - {iv.get('target', '_')}))")
        elif t == "cross_entropy_loss":
            call_lines.append(f"{dst} = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels={iv.get('target', '_')}, logits={iv.get('pred', '_')}))")
        elif t == "l1_loss":
            call_lines.append(f"{dst} = tf.reduce_mean(tf.abs({iv.get('pred', '_')} - {iv.get('target', '_')}))")
        elif t == "binary_cross_entropy":
            call_lines.append(f"{dst} = tf.reduce_mean(tf.keras.losses.binary_crossentropy({iv.get('target', '_')}, {iv.get('pred', '_')}))")

        # ---------------- fallback ----------------
        else:
            call_lines.append(f"{dst} = None  # 无 TF 等价映射: '{t}'")

    ret = returns[-1] if len(returns) == 1 else ("(" + ", ".join(returns) + ")" if returns else (last_var or "None"))

    init_code = "\n".join(f"        {line}" for line in init_lines) if init_lines else "        pass"
    call_code = "\n".join(f"        {line}" for line in call_lines) if call_lines else "        pass"
    args = ", ".join(params)

    return (
        '"""Generated by Entropia Riko (TensorFlow/Keras backend)."""\n\n'
        "import tensorflow as tf\n\n\n"
        "class GraphModel(tf.keras.Model):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        f"{init_code}\n\n"
        f"    def call(self, {args}):\n"
        f"{call_code}\n"
        f"        return {ret}\n"
    )
