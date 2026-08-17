"""Export a graph document as a clean, runnable PyTorch ``nn.Module`` script.

The output follows the standard PyTorch module pattern::

    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class GraphModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)

        def forward(self, input):
            input = self.fc1(input)
            input = torch.relu(input)
            return input

Conventions:

- ``graph_input`` nodes become ``forward`` parameters; ``graph_output`` becomes
  the ``return`` statement.
- Learnable layers (linear / conv / transformer / ...) are declared in
  ``__init__``; inline ops run in ``forward``.
- ``graph_reference`` / ``import`` nodes are inlined recursively as nested
  ``nn.Module`` classes, so an importing graph exports to a single file.
- Data loaders generate real, runnable loading code (torchvision with a
  synthetic-data fallback), so a self-contained graph (data loader + loss) is
  actually trainable: a ``train()`` helper and a ``__main__`` block are emitted.
- Variable names are sanitized (keyword / reserved-word safe). Input ports are
  resolved robustly: if an edge references a stale port name, the node's single
  output is used as a fallback instead of emitting a broken ``_`` placeholder.
"""
from __future__ import annotations

import json
import keyword
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.document import GraphDocument
from .executor import execution_order
from .registry import Registry, default_registry
from .subgraph import resolve_graph_file

_RESERVED = {"self", "torch", "nn", "F", "GraphModel", "model", "math", "input"}


def _sanitize(name: Any) -> str:
    """Turn a node id / name into a valid, non-reserved Python identifier."""
    s = re.sub(r"\W+", "_", str(name)).strip("_")
    if not s:
        s = "node"
    if s[0].isdigit():
        s = "n_" + s
    if keyword.iskeyword(s) or s in _RESERVED:
        s = s + "_"
    return s


def _pascal(name: Any) -> str:
    """PascalCase for nested nn.Module class names."""
    parts = re.split(r"[_\W]+", str(name))
    return "".join(p.capitalize() for p in parts if p) or "Subgraph"


def _lit(value: Any) -> str:
    """Python literal for a scalar / nested-list payload."""
    return repr(value)


def _tuple_lit(value: Any) -> str:
    """Tuple literal for shape / dims (list [3, 4] -> '(3, 4)')."""
    if isinstance(value, (list, tuple)):
        return repr(tuple(value))
    return repr(value)


def _dim(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return repr(tuple(value))
    return str(int(value))


# Node types that become nn.Module submodules declared in __init__.
_NN_LAYERS = {
    "linear", "conv2d", "conv1d", "conv_transpose2d", "maxpool2d",
    "avgpool2d", "embedding", "dropout", "batchnorm1d", "layernorm",
    "transformer_encoder", "batchnorm2d", "groupnorm", "rmsnorm",
    "multihead_attention",
}

_LOSS_TYPES = {
    "mse_loss", "cross_entropy_loss", "l1_loss", "smooth_l1_loss",
    "binary_cross_entropy", "kl_div", "nll_loss", "hinge_embedding_loss",
    "cosine_embedding_loss",
}


def _nn_ctor(t: str, p: Dict[str, Any]) -> str:
    """Return the nn.<Layer>(...) constructor expression (no 'self.')."""
    if t == "linear":
        return (f"nn.Linear({p.get('in_features')}, {p.get('out_features')}, "
                f"bias={p.get('use_bias', True)})")
    if t in ("conv2d", "conv1d", "conv_transpose2d"):
        cls = {"conv2d": "Conv2d", "conv1d": "Conv1d",
               "conv_transpose2d": "ConvTranspose2d"}[t]
        return (f"nn.{cls}({p.get('in_channels')}, {p.get('out_channels')}, "
                f"{p.get('kernel_size')}, stride={p.get('stride', 1)}, "
                f"padding={p.get('padding', 0)})")
    if t == "maxpool2d":
        ks = p.get("kernel_size")
        stride = p.get("stride") or ks
        return f"nn.MaxPool2d({ks}, stride={stride})"
    if t == "avgpool2d":
        ks = p.get("kernel_size")
        stride = p.get("stride") or ks
        return f"nn.AvgPool2d({ks}, stride={stride})"
    if t == "embedding":
        return f"nn.Embedding({p.get('num_embeddings')}, {p.get('embedding_dim')})"
    if t == "dropout":
        return f"nn.Dropout({p.get('p', 0.5)})"
    if t == "batchnorm1d":
        return f"nn.BatchNorm1d({p.get('num_features')})"
    if t == "layernorm":
        shape = p.get("normalized_shape")
        shape_lit = _tuple_lit(shape) if isinstance(shape, (list, tuple)) else str(int(shape))
        return f"nn.LayerNorm({shape_lit})"
    if t == "transformer_encoder":
        dm = p.get("d_model", 64)
        df = p.get("dim_feedforward", dm * 4)
        return (f"nn.TransformerEncoder(nn.TransformerEncoderLayer("
                f"d_model={dm}, nhead={p.get('nhead')}, "
                f"dim_feedforward={df}, batch_first={p.get('batch_first', True)}), "
                f"num_layers={p.get('num_layers', 1)})")
    if t == "batchnorm2d":
        return f"nn.BatchNorm2d({p.get('num_features')})"
    if t == "groupnorm":
        return f"nn.GroupNorm({p.get('num_groups')}, {p.get('num_channels')})"
    if t == "rmsnorm":
        return f"RMSNorm({p.get('dim')}, eps={p.get('eps', 1e-6)})"
    if t == "multihead_attention":
        return f"nn.MultiheadAttention({p.get('embed_dim')}, {p.get('num_heads')}, batch_first=True)"
    raise ValueError(f"未处理的 NN 层类型 {t}")


def _nn_input(t: str, iv: Dict[str, str]) -> str:
    """Forward input expression for an nn layer node."""
    if t == "embedding":
        return f"{iv.get('indices', '_')}.long()"
    return iv.get("x", "_")


def _nn_extra_init(t: str, p: Dict[str, Any], var: str) -> List[str]:
    """Optional weight/bias override lines (keep codegen faithful to runtime)."""
    lines: List[str] = []
    if p.get("weight") is not None:
        lines.append(
            f"{var}.weight.data = torch.tensor({_lit(p['weight'])}, dtype=torch.float32)"
        )
    if p.get("bias") is not None:
        lines.append(
            f"{var}.bias.data = torch.tensor({_lit(p['bias'])}, dtype=torch.float32)"
        )
    return lines


class _Emitter:
    """Recursive code generator. Inlines imported graphs as nested modules."""

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.nested_classes: List[str] = []
        self.helpers: List[str] = []
        self._helper_names: Set[str] = set()
        self._generated: Dict[str, str] = {}   # resolved path -> class name
        self._in_progress: Set[str] = set()
        self._class_names: Set[str] = set()

    def _class_name(self, hint: Any) -> str:
        base = _pascal(hint)
        name = base
        i = 1
        while name in self._class_names or name == "GraphModel":
            i += 1
            name = f"{base}{i}"
        self._class_names.add(name)
        return name

    def _add_helper(self, name: str, lines: List[str]) -> str:
        """Register a module-level helper function (dedupe by name)."""
        if name not in self._helper_names:
            self._helper_names.add(name)
            self.helpers.append("\n".join(lines))
        return name

    @staticmethod
    def _source_var(node: str, port: str, node_vars: Dict[str, Dict[str, str]]) -> str:
        """Resolve an edge source to a variable name, robust to stale port names."""
        ports = node_vars.get(node)
        if not ports:
            return "_"
        if port in ports:
            return ports[port]
        if len(ports) == 1:
            return next(iter(ports.values()))
        return "_"

    def _render(self, class_name: str, init: List[str], fwd: List[str],
                params: List[str], ret: str) -> str:
        args = ", ".join(params)
        sig = f"def forward(self{', ' + args if args else ''}):"
        lines = [
            f"class {class_name}(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
        ]
        if init:
            lines += [f"        {l}" for l in init]
        else:
            lines.append("        pass")
        lines.append("")
        lines.append(f"    {sig}")
        if fwd:
            lines += [f"        {l}" for l in fwd]
        else:
            lines.append("        pass")
        lines.append(f"        return {ret}")
        return "\n".join(lines)

    def _generate(self, doc: GraphDocument, class_name: str) -> Tuple[List[str], List[str], List[str], str]:
        reg = self.registry
        try:
            order = execution_order(doc)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"图 '{class_name}' 拓扑排序失败: {exc}") from exc

        # 1. forward parameters come from graph_input nodes.
        params: List[str] = []
        param_seen: Set[str] = set()
        for nid in order:
            node = doc.nodes[nid]
            if node.type_name != "graph_input":
                continue
            base = _sanitize(node.parameters.get("name", "x"))
            name = base
            i = 1
            while name in param_seen:
                i += 1
                name = f"{base}_{i}"
            param_seen.add(name)
            params.append(name)

        # 2. assign a variable name to every node output.
        used: Set[str] = set(params) | set(self._class_names)

        def _unique(base: str) -> str:
            name = base
            i = 1
            while name in used:
                i += 1
                name = f"{base}_{i}"
            used.add(name)
            return name

        out_vars: Dict[Tuple[str, str], str] = {}
        node_vars: Dict[str, Dict[str, str]] = {}

        graph_input_idx = 0
        for nid in order:
            node = doc.nodes[nid]
            t = node.type_name
            if t == "graph_input":
                out_vars[(nid, "value")] = params[graph_input_idx]
                node_vars[nid] = {"value": params[graph_input_idx]}
                graph_input_idx += 1
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

        # 3. resolve incoming edges.
        in_edges: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for e in doc.edges:
            in_edges[(e.target_node, e.target_port)] = (e.source_node, e.source_port)

        init_lines: List[str] = []
        forward_lines: List[str] = []
        returns: List[str] = []
        last_var: Optional[str] = None

        for nid in order:
            node = doc.nodes[nid]
            t = node.type_name
            p = dict(node.parameters)
            cls = reg.get(t) if t in reg else None

            # input variables (robust to stale source port names)
            iv: Dict[str, str] = {}
            if cls:
                for inp in cls.inputs:
                    key = (nid, inp.name)
                    if key in in_edges:
                        src_node, src_port = in_edges[key]
                        iv[inp.name] = self._source_var(src_node, src_port, node_vars)
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

            # ---------------- inline tensor / activation ops ----------------
            if t in ("abs", "exp", "log", "sqrt", "sign", "reciprocal",
                     "floor", "ceil", "round", "cos", "sin",
                     "log2", "log10", "log1p", "expm1", "erf", "erfc"):
                fn = {"abs": "torch.abs", "exp": "torch.exp", "log": "torch.log",
                      "sqrt": "torch.sqrt", "sign": "torch.sign",
                      "reciprocal": "torch.reciprocal", "floor": "torch.floor",
                      "ceil": "torch.ceil", "round": "torch.round",
                      "cos": "torch.cos", "sin": "torch.sin",
                      "log2": "torch.log2", "log10": "torch.log10",
                      "log1p": "torch.log1p", "expm1": "torch.expm1",
                      "erf": "torch.erf", "erfc": "torch.erfc"}[t]
                forward_lines.append(f"{dst} = {fn}({iv.get('x', '_')})")
            elif t == "neg":
                forward_lines.append(f"{dst} = -{iv.get('x', '_')}")
            elif t == "square":
                forward_lines.append(f"{dst} = {iv.get('x', '_')} * {iv.get('x', '_')}")
            elif t in ("relu", "sigmoid", "tanh", "gelu", "silu", "selu", "elu",
                       "mish", "hardswish", "softplus", "relu6",
                       "softsign", "tanhshrink", "hardsigmoid", "log_sigmoid"):
                fn = {"relu": "torch.relu", "sigmoid": "torch.sigmoid",
                      "tanh": "torch.tanh", "gelu": "F.gelu", "silu": "F.silu",
                      "selu": "F.selu", "elu": "F.elu", "mish": "F.mish",
                      "hardswish": "F.hardswish", "softplus": "F.softplus",
                      "relu6": "F.relu6", "softsign": "F.softsign",
                      "tanhshrink": "F.tanhshrink", "hardsigmoid": "F.hardsigmoid",
                      "log_sigmoid": "F.logsigmoid"}[t]
                forward_lines.append(f"{dst} = {fn}({iv.get('x', '_')})")
            elif t == "leaky_relu":
                forward_lines.append(f"{dst} = F.leaky_relu({iv.get('x', '_')}, negative_slope={p.get('negative_slope', 0.01)})")
            elif t == "hardtanh":
                forward_lines.append(f"{dst} = F.hardtanh({iv.get('x', '_')}, min_val={p.get('min_val', -1.0)}, max_val={p.get('max_val', 1.0)})")
            elif t == "glu":
                forward_lines.append(f"{dst} = F.glu({iv.get('x', '_')}, dim={p.get('dim', -1)})")
            elif t == "softmax":
                forward_lines.append(f"{dst} = F.softmax({iv.get('x', '_')}, dim={p.get('dim', -1)})")
            elif t == "log_softmax":
                forward_lines.append(f"{dst} = F.log_softmax({iv.get('x', '_')}, dim={p.get('dim', -1)})")

            # ---------------- binary math ----------------
            elif t in ("add", "multiply", "sub", "div", "pow", "matmul", "mm",
                       "maximum", "minimum", "fmod", "remainder", "atan2",
                       "torch_add", "torch_multiply", "bmm", "dot", "outer", "xlogy"):
                a, b = iv.get("left", "_"), iv.get("right", "_")
                ops = {"add": "+", "multiply": "*", "sub": "-",
                       "div": "torch.div", "pow": "torch.pow",
                       "matmul": "torch.matmul", "mm": "torch.mm",
                       "maximum": "torch.maximum", "minimum": "torch.minimum",
                       "fmod": "torch.fmod", "remainder": "torch.remainder",
                       "atan2": "torch.atan2", "torch_add": "torch.add",
                       "torch_multiply": "torch.mul", "bmm": "torch.bmm",
                       "dot": "torch.dot", "outer": "torch.outer", "xlogy": "torch.xlogy"}
                op = ops[t]
                if op in ("+", "-", "*"):
                    forward_lines.append(f"{dst} = {a} {op} {b}")
                else:
                    forward_lines.append(f"{dst} = {op}({a}, {b})")
            elif t == "cross":
                forward_lines.append(
                    f"{dst} = torch.cross({iv.get('left', '_')}, {iv.get('right', '_')}, dim={p.get('dim', -1)})"
                )
            elif t == "addmm":
                forward_lines.append(
                    f"{dst} = torch.addmm({iv.get('input', '_')}, {iv.get('mat1', '_')}, "
                    f"{iv.get('mat2', '_')}, beta={p.get('beta', 1.0)}, alpha={p.get('alpha', 1.0)})"
                )

            # ---------------- reduce ----------------
            elif t in ("sum", "mean", "prod", "std", "var"):
                x = iv.get("x", "_")
                m = {"sum": "sum", "mean": "mean", "prod": "prod",
                     "std": "std", "var": "var"}[t]
                if p.get("dim") is None:
                    forward_lines.append(f"{dst} = {x}.{m}()")
                else:
                    args = f"dim={_dim(p['dim'])}"
                    if p.get("keepdim") is not None:
                        args += f", keepdim={bool(p['keepdim'])}"
                    forward_lines.append(f"{dst} = {x}.{m}({args})")
            elif t in ("amax", "amin", "logsumexp"):
                x = iv.get("x", "_")
                fn = {"amax": "torch.amax", "amin": "torch.amin", "logsumexp": "torch.logsumexp"}[t]
                if p.get("dim") is None:
                    forward_lines.append(f"{dst} = {fn}({x})")
                else:
                    args = f"dim={_dim(p['dim'])}"
                    if p.get("keepdim") is not None:
                        args += f", keepdim={bool(p['keepdim'])}"
                    forward_lines.append(f"{dst} = {fn}({x}, {args})")
            elif t == "median":
                x = iv.get("x", "_")
                if p.get("dim") is None:
                    forward_lines.append(f"{dst} = torch.median({x})")
                else:
                    args = f"dim={_dim(p['dim'])}"
                    if p.get("keepdim") is not None:
                        args += f", keepdim={bool(p['keepdim'])}"
                    forward_lines.append(f"{dst} = torch.median({x}, {args}).values")
            elif t in ("max_reduce", "min_reduce"):
                fn = "torch.max" if t == "max_reduce" else "torch.min"
                x = iv.get("x", "_")
                if p.get("dim") is None:
                    forward_lines.append(f"{dst} = {fn}({x})")
                else:
                    args = f"dim={_dim(p['dim'])}"
                    if p.get("keepdim") is not None:
                        args += f", keepdim={bool(p['keepdim'])}"
                    forward_lines.append(f"{dst} = {fn}({x}, {args}).values")
            elif t == "norm":
                x = iv.get("x", "_")
                if p.get("dim") is None:
                    forward_lines.append(f"{dst} = torch.norm({x}.float())")
                else:
                    args = f"dim={_dim(p['dim'])}"
                    if p.get("keepdim") is not None:
                        args += f", keepdim={bool(p['keepdim'])}"
                    forward_lines.append(f"{dst} = torch.norm({x}.float(), {args})")
            elif t == "argmax":
                x = iv.get("x", "_")
                forward_lines.append(f"{dst} = {x}.argmax()" if p.get("dim") is None
                                     else f"{dst} = {x}.argmax(dim={_dim(p['dim'])})")
            elif t == "argmin":
                x = iv.get("x", "_")
                forward_lines.append(f"{dst} = {x}.argmin()" if p.get("dim") is None
                                     else f"{dst} = {x}.argmin(dim={_dim(p['dim'])})")
            elif t == "cumsum":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.cumsum(dim={_dim(p['dim'])})")
            elif t == "topk":
                forward_lines.append(
                    f"{dst} = {iv.get('x', '_')}.topk({p.get('k')}, dim={p.get('dim', -1)}).values"
                )
            elif t == "sort":
                forward_lines.append(
                    f"{dst} = {iv.get('x', '_')}.sort(dim={p.get('dim', -1)}, "
                    f"descending={bool(p.get('descending', False))}).values"
                )

            # ---------------- shape ----------------
            elif t == "reshape":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.reshape({_tuple_lit(p['shape'])})")
            elif t == "transpose":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.transpose({p.get('dim0')}, {p.get('dim1')})")
            elif t == "permute":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.permute({_tuple_lit(p['dims'])})")
            elif t == "flatten":
                forward_lines.append(
                    f"{dst} = {iv.get('x', '_')}.flatten({p.get('start_dim', 0)}, {p.get('end_dim', -1)})"
                )
            elif t == "squeeze":
                x = iv.get("x", "_")
                forward_lines.append(f"{dst} = {x}.squeeze()" if p.get("dim") is None
                                     else f"{dst} = {x}.squeeze({_dim(p['dim'])})")
            elif t == "unsqueeze":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.unsqueeze({_dim(p['dim'])})")
            elif t == "concat":
                forward_lines.append(
                    f"{dst} = torch.cat([{iv.get('left', '_')}, {iv.get('right', '_')}], dim={_dim(p['dim'])})"
                )
            elif t == "stack":
                forward_lines.append(
                    f"{dst} = torch.stack([{iv.get('left', '_')}, {iv.get('right', '_')}], dim={_dim(p['dim'])})"
                )
            elif t == "flip":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.flip({_dim(p['dims'])})")
            elif t == "expand_as":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.expand_as({iv.get('other', '_')})")
            elif t == "repeat_interleave":
                x = iv.get("x", "_")
                dim = p.get("dim")
                expr = f"{x}.repeat_interleave({p.get('repeats')}" + (f", dim={_dim(dim)}" if dim is not None else "") + ")"
                forward_lines.append(f"{dst} = {expr}")
            elif t == "view":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.view({_tuple_lit(p['shape'])})")
            elif t == "swapaxes":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.swapaxes({p.get('dim0')}, {p.get('dim1')})")
            elif t == "movedim":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.movedim({p.get('source')}, {p.get('destination')})")
            elif t == "expand":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.expand({_tuple_lit(p['shape'])})")
            elif t == "broadcast_to":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.broadcast_to({_tuple_lit(p['shape'])})")
            elif t in ("tile", "repeat"):
                m = "tile" if t == "tile" else "repeat"
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.{m}({_tuple_lit(p['dims'])})")
            elif t == "tril":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.tril({p.get('diagonal', 0)})")
            elif t == "triu":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.triu({p.get('diagonal', 0)})")
            elif t == "diagonal":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.diagonal({p.get('offset', 0)}, {p.get('dim1', 0)}, {p.get('dim2', 1)})")
            elif t == "narrow":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.narrow({p.get('dim')}, {p.get('start')}, {p.get('length')})")
            elif t == "roll":
                x = iv.get("x", "_")
                shifts = _dim(p.get("shifts"))
                dims = p.get("dims")
                expr = f"{x}.roll({shifts}" + (f", {_dim(dims)}" if dims is not None else "") + ")"
                forward_lines.append(f"{dst} = {expr}")
            elif t == "interpolate":
                size = _tuple_lit(p["size"]) if p.get("size") is not None else "None"
                scale = repr(p["scale_factor"]) if p.get("scale_factor") is not None else "None"
                forward_lines.append(f"{dst} = F.interpolate({iv.get('x', '_')}, size={size}, scale_factor={scale}, mode={_lit(p.get('mode', 'nearest'))})")
            elif t == "index_select":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.index_select({p.get('dim')}, {iv.get('index', '_')}.long())")
            elif t == "gather":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.gather({p.get('dim')}, {iv.get('index', '_')}.long())")
            elif t == "einsum":
                forward_lines.append(f"{dst} = torch.einsum({_lit(p.get('equation'))}, {iv.get('a', '_')}, {iv.get('b', '_')})")
            elif t == "sdpa":
                forward_lines.append(
                    f"{dst} = F.scaled_dot_product_attention({iv.get('q', '_')}, {iv.get('k', '_')}, "
                    f"{iv.get('v', '_')}, is_causal={bool(p.get('is_causal', False))})"
                )

            # ---------------- creation ----------------
            elif t in ("zeros", "ones", "rand", "randn"):
                fn = {"zeros": "torch.zeros", "ones": "torch.ones",
                      "rand": "torch.rand", "randn": "torch.randn"}[t]
                forward_lines.append(f"{dst} = {fn}({_tuple_lit(p['shape'])})")
            elif t == "eye":
                forward_lines.append(f"{dst} = torch.eye({p.get('n')})")
            elif t == "arange":
                forward_lines.append(
                    f"{dst} = torch.arange({p.get('start', 0)}, {p.get('end')}, {p.get('step', 1)})"
                )
            elif t == "linspace":
                forward_lines.append(
                    f"{dst} = torch.linspace({p.get('start')}, {p.get('end')}, {p.get('steps')})"
                )
            elif t == "full":
                forward_lines.append(f"{dst} = torch.full({_tuple_lit(p['shape'])}, {p.get('fill_value')})")
            elif t == "randint":
                forward_lines.append(f"{dst} = torch.randint({p.get('low', 0)}, {p.get('high')}, {_tuple_lit(p['size'])})")
            elif t == "randperm":
                forward_lines.append(f"{dst} = torch.randperm({p.get('n')})")
            elif t == "empty":
                forward_lines.append(f"{dst} = torch.empty({_tuple_lit(p['shape'])})")
            elif t in ("zeros_like", "ones_like", "randn_like", "rand_like"):
                fn = {"zeros_like": "torch.zeros_like", "ones_like": "torch.ones_like",
                      "randn_like": "torch.randn_like", "rand_like": "torch.rand_like"}[t]
                forward_lines.append(f"{dst} = {fn}({iv.get('x', '_')})")
            elif t == "positional_encoding":
                self._add_helper("_sincos_1d", [
                    "def _sincos_1d(n, dim):",
                    "    pos = torch.arange(n).unsqueeze(1).float()",
                    "    div = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))",
                    "    pe = torch.zeros(n, dim)",
                    "    pe[:, 0::2] = torch.sin(pos * div)",
                    "    pe[:, 1::2] = torch.cos(pos * div)",
                    "    return pe",
                ])
                forward_lines.append(f"{dst} = _sincos_1d({p.get('n')}, {p.get('dim')})")

            # ---------------- device / tensor utils ----------------
            elif t == "to_device":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.to('{p.get('device', 'cpu')}')")
            elif t == "contiguous":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.contiguous()")
            elif t == "clone":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.clone()")
            elif t == "detach":
                forward_lines.append(f"{dst} = {iv.get('x', '_')}.detach()")
            elif t == "clamp":
                x = iv.get("x", "_")
                mn, mx = p.get("min"), p.get("max")
                if mn is None and mx is None:
                    forward_lines.append(f"{dst} = {x}.clamp()")
                else:
                    kwargs = []
                    if mn is not None:
                        kwargs.append(f"min={_lit(mn)}")
                    if mx is not None:
                        kwargs.append(f"max={_lit(mx)}")
                    forward_lines.append(f"{dst} = {x}.clamp({', '.join(kwargs)})")
            elif t == "where":
                forward_lines.append(
                    f"{dst} = torch.where({iv.get('condition', '_')}.bool(), "
                    f"{iv.get('x', '_')}, {iv.get('y', '_')})"
                )
            elif t == "masked_fill":
                forward_lines.append(
                    f"{dst} = {iv.get('x', '_')}.masked_fill({iv.get('mask', '_')}.bool(), {p.get('value')})"
                )

            # ---------------- NN layers (declared in __init__) ----------------
            elif t in _NN_LAYERS:
                if t == "rmsnorm":
                    self._add_helper("RMSNorm", [
                        "class RMSNorm(nn.Module):",
                        "    def __init__(self, dim, eps=1e-6):",
                        "        super().__init__()",
                        "        self.weight = nn.Parameter(torch.ones(dim))",
                        "        self.eps = eps",
                        "    def forward(self, x):",
                        "        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()",
                        "        return x / rms * self.weight",
                    ])
                init_lines.append(f"self.{dst} = {_nn_ctor(t, p)}")
                init_lines += _nn_extra_init(t, p, f"self.{dst}")
                if t == "multihead_attention":
                    x = iv.get("x", "_")
                    forward_lines.append(f"{dst} = self.{dst}({x}, {x}, {x}, need_weights=False)[0]")
                else:
                    forward_lines.append(f"{dst} = self.{dst}({_nn_input(t, iv)})")

            # ---------------- constants / losses / models ----------------
            elif t == "constant":
                forward_lines.append(f"{dst} = torch.tensor({_lit(p.get('value'))}, dtype=torch.float32)")
            elif t in _LOSS_TYPES:
                fn = {
                    "mse_loss": "F.mse_loss", "cross_entropy_loss": "F.cross_entropy",
                    "l1_loss": "F.l1_loss", "smooth_l1_loss": "F.smooth_l1_loss",
                    "binary_cross_entropy": "F.binary_cross_entropy", "kl_div": "F.kl_div",
                    "nll_loss": "F.nll_loss", "hinge_embedding_loss": "F.hinge_embedding_loss",
                    "cosine_embedding_loss": "F.cosine_embedding_loss",
                }[t]
                target = iv.get("target", "_")
                if t in ("cross_entropy_loss", "nll_loss"):
                    target = f"{target}.long()"
                forward_lines.append(f"{dst} = {fn}({iv.get('pred', '_')}, {target})")
            elif t == "inference":
                forward_lines.append(f"{dst} = {iv.get('model', '_')}({iv.get('x', '_')})")

            # ---------------- data loaders (real, runnable code) ----------------
            elif t in ("mnist_loader", "cifar10_loader"):
                kind = "MNIST" if t == "mnist_loader" else "CIFAR10"
                ch = 1 if t == "mnist_loader" else 3
                size = 28 if t == "mnist_loader" else 32
                ncls = 10
                helper = self._add_helper(
                    f"_load_{_sanitize(nid)}",
                    [
                        f"def _load_{_sanitize(nid)}(batch_size={p.get('batch_size', 32)}, train={p.get('train', True)}, device={_lit(p.get('device', 'cpu'))}):",
                        "    try:",
                        "        import torchvision",
                        "        import torchvision.transforms as transforms",
                        f"        ds = torchvision.datasets.{kind}(root=\"./data\", train=train, download=True, transform=transforms.ToTensor())",
                        "        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train)",
                        "        images, labels = next(iter(loader))",
                        "        return images.to(device), labels.to(device)",
                        "    except Exception:",
                        f"        return torch.rand(batch_size, {ch}, {size}, {size}, device=device), torch.randint(0, {ncls}, (batch_size,), device=device)",
                    ],
                )
                imgs = out_vars.get((nid, "images"))
                labels = out_vars.get((nid, "labels"))
                forward_lines.append(f"{imgs}, {labels} = {helper}()")
            elif t == "csv_loader":
                helper = self._add_helper(
                    f"_load_{_sanitize(nid)}",
                    [
                        f"def _load_{_sanitize(nid)}(path, delimiter={_lit(p.get('delimiter', ','))}, skip={int(p.get('skip_header', 1))}):",
                        "    import csv",
                        "    rows = []",
                        "    with open(path, \"r\", encoding=\"utf-8\") as f:",
                        "        reader = csv.reader(f, delimiter=delimiter)",
                        "        for _ in range(skip):",
                        "            next(reader, None)",
                        "        for row in reader:",
                        "            rows.append([float(v) for v in row])",
                        "    if not rows:",
                        "        raise ValueError(\"CSV 无数据: \" + path)",
                        "    return torch.tensor(rows, dtype=torch.float32)",
                    ],
                )
                forward_lines.append(f"{dst} = {helper}({_lit(p.get('path'))})")
            elif t == "tensor_file_loader":
                forward_lines.append(
                    f"{dst} = torch.load({_lit(p.get('path'))}, map_location={_lit(p.get('device', 'cpu'))}, weights_only=True)"
                )
            elif t == "image_folder_loader":
                h = int(p.get("height", 224))
                w = int(p.get("width", 224))
                helper = self._add_helper(
                    f"_load_{_sanitize(nid)}",
                    [
                        f"def _load_{_sanitize(nid)}(path, batch_size={p.get('batch_size', 8)}, height={h}, width={w}, device={_lit(p.get('device', 'cpu'))}):",
                        "    try:",
                        "        import torchvision",
                        "        import torchvision.transforms as transforms",
                        "        transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])",
                        "        ds = torchvision.datasets.ImageFolder(path, transform=transform)",
                        "        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)",
                        "        images, _ = next(iter(loader))",
                        "        return images.to(device)",
                        "    except Exception:",
                        "        return torch.rand(batch_size, 3, height, width, device=device)",
                    ],
                )
                forward_lines.append(f"{dst} = {helper}({_lit(p.get('path'))})")
            elif t == "dataloader":
                forward_lines.append(
                    f"{dst} = torch.rand({p.get('batch_size', 32)}, {p.get('channels', 1)}, "
                    f"{p.get('height', 28)}, {p.get('width', 28)})"
                )
            elif t == "model_loader":
                helper = self._add_helper(
                    f"_load_model_{_sanitize(nid)}",
                    [
                        f"def _load_model_{_sanitize(nid)}(path, template=None, device={_lit(p.get('device', 'cpu'))}):",
                        "    import torch",
                        "    from pathlib import Path",
                        "    path = Path(path).expanduser()",
                        "    if not path.exists():",
                        "        raise FileNotFoundError(f'模型文件不存在: {path}')",
                        "    if path.suffix.lower() == '.safetensors':",
                        "        from safetensors.torch import load_file",
                        "        state = load_file(str(path), device=device)",
                        "        if template is not None and hasattr(template, 'load_state_dict'):",
                        "            template.load_state_dict(state)",
                        "            return template",
                        "        return state",
                        "    obj = torch.load(str(path), map_location=device, weights_only=False)",
                        "    if isinstance(obj, dict) and template is not None and hasattr(template, 'load_state_dict'):",
                        "        template.load_state_dict(obj)",
                        "        return template",
                        "    return obj",
                    ],
                )
                forward_lines.append(
                    f"{dst} = _load_model_{_sanitize(nid)}({_lit(p.get('path'))}, template={iv.get('template', 'None')})"
                )
            elif t == "save_model":
                helper = self._add_helper(
                    f"_save_model_{_sanitize(nid)}",
                    [
                        f"def _save_model_{_sanitize(nid)}(path, model, fmt='auto'):",
                        "    import torch",
                        "    from pathlib import Path",
                        "    path = Path(path).expanduser()",
                        "    path.parent.mkdir(parents=True, exist_ok=True)",
                        "    if path.suffix.lower() == '.safetensors' and fmt in ('auto', 'safetensors'):",
                        "        from safetensors.torch import save_file",
                        "        state = model.state_dict() if hasattr(model, 'state_dict') else model",
                        "        save_file(state, str(path))",
                        "    else:",
                        "        torch.save(model, str(path))",
                        "    return str(path)",
                    ],
                )
                forward_lines.append(
                    f"{dst} = _save_model_{_sanitize(nid)}({_lit(p.get('path'))}, {iv.get('model', 'None')}, {_lit(p.get('format', 'auto'))})"
                )
            elif t == "text_loader":
                forward_lines.append(f"{dst} = open({_lit(p.get('path'))}, encoding=\"utf-8\").read()")
            elif t == "json_loader":
                helper = self._add_helper(
                    f"_load_json_{_sanitize(nid)}",
                    [
                        f"def _load_json_{_sanitize(nid)}(path):",
                        "    import json",
                        "    with open(path, encoding=\"utf-8\") as f:",
                        "        return json.load(f)",
                    ],
                )
                forward_lines.append(f"{dst} = {helper}({_lit(p.get('path'))})")
            elif t == "image_loader":
                h = int(p.get("height", 0))
                w = int(p.get("width", 0))
                helper = self._add_helper(
                    f"_load_image_{_sanitize(nid)}",
                    [
                        f"def _load_image_{_sanitize(nid)}(path, height={h}, width={w}):",
                        "    try:",
                        "        from PIL import Image",
                        "        img = Image.open(path).convert(\"RGB\")",
                        "        if height and width:",
                        "            img = img.resize((width, height))",
                        "        arr = torch.tensor(list(img.getdata()), dtype=torch.float32)",
                        "        return arr.view(img.size[1], img.size[0], 3) / 255.0",
                        "    except Exception:",
                        "        return torch.rand(height or 64, width or 64, 3)",
                    ],
                )
                forward_lines.append(f"{dst} = {helper}({_lit(p.get('path'))})")

            # ---------------- subgraph reference / import (recursive inline) ----
            elif t in ("graph_reference", "import"):
                spec = p.get("file") if t == "graph_reference" else p.get("module")
                path = resolve_graph_file(spec) if spec else None
                if path is None:
                    if dst is not None:
                        forward_lines.append(f"{dst} = None  # {t} '{spec}' not found")
                    continue
                key = str(path)
                if key in self._in_progress:
                    if dst is not None:
                        forward_lines.append(f"{dst} = None  # cyclic import of '{spec}' not inlined")
                    continue
                if key in self._generated:
                    sub_cls = self._generated[key]
                else:
                    self._in_progress.add(key)
                    try:
                        sub_cls = self._class_name(path.stem)
                        sub_doc = load_graph_file_from_path(path)
                        s_init, s_fwd, s_params, s_ret = self._generate(sub_doc, sub_cls)
                        self.nested_classes.append(self._render(sub_cls, s_init, s_fwd, s_params, s_ret))
                    finally:
                        self._in_progress.discard(key)
                    self._generated[key] = sub_cls
                if dst is not None:
                    init_lines.append(f"self.{dst} = {sub_cls}()")
                    forward_lines.append(f"{dst} = self.{dst}({iv.get('input', '_')})")

            # ---------------- wrangle (inline Python) + previews ----------------
            elif t == "wrangle":
                for name in ("x", "a", "b"):
                    forward_lines.append(f"{name} = {iv[name] if name in iv else 'None'}")
                for line in str(p.get("code", "")).splitlines():
                    forward_lines.append(line)
                forward_lines.append(f"{dst} = result")
            elif t in ("text_preview", "json_preview", "image_preview"):
                src = iv.get("x", "_")
                if t == "text_preview":
                    forward_lines.append(f"{dst} = str({src})")
                elif t == "json_preview":
                    helper = self._add_helper("_json_dumps", [
                        "def _json_dumps(x, indent=2):",
                        "    import json",
                        "    return json.dumps(x, ensure_ascii=False, indent=indent)",
                    ])
                    forward_lines.append(f"{dst} = {helper}({src})")
                else:  # image_preview: pass-through
                    forward_lines.append(f"{dst} = {src}")

            # ---------------- Hugging Face pretrained models ----------------
            elif t == "diffusers_text2img":
                h = int(p.get("height", 512))
                w = int(p.get("width", 512))
                helper = self._add_helper(f"_diffusers_{_sanitize(nid)}", [
                    f"def _diffusers_{_sanitize(nid)}(model_id, prompt, num_steps={p.get('num_steps', 4)}, guidance_scale={p.get('guidance_scale', 0.0)}, height={h}, width={w}):",
                    "    try:",
                    "        from diffusers import DiffusionPipeline",
                    "        pipe = DiffusionPipeline.from_pretrained(model_id)",
                    "        img = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale, height=height, width=width).images[0].convert(\"RGB\")",
                    "        return torch.tensor(list(img.getdata()), dtype=torch.float32).view(img.size[1], img.size[0], 3) / 255.0",
                    "    except Exception:",
                    "        return torch.rand(height, width, 3)",
                ])
                forward_lines.append(f"{dst} = {helper}({_lit(p.get('model_id'))}, {_lit(p.get('prompt'))})")
            elif t == "transformers_pipeline":
                helper = self._add_helper(f"_hf_pipe_{_sanitize(nid)}", [
                    f"def _hf_pipe_{_sanitize(nid)}(task, model_id, text):",
                    "    import json",
                    "    try:",
                    "        from transformers import pipeline",
                    "        return json.dumps(pipeline(task, model=model_id)(text), ensure_ascii=False, indent=2)",
                    "    except Exception as e:",
                    "        return f\"(transformers failed) {e}\"",
                ])
                forward_lines.append(f"{dst} = {helper}({_lit(p.get('task'))}, {_lit(p.get('model_id'))}, {_lit(p.get('text'))})")
            elif t == "transformers_embedding":
                helper = self._add_helper(f"_hf_emb_{_sanitize(nid)}", [
                    f"def _hf_emb_{_sanitize(nid)}(model_id, text, pool={_lit(p.get('pool', 'mean'))}):",
                    "    try:",
                    "        from transformers import AutoModel, AutoTokenizer",
                    "        tok = AutoTokenizer.from_pretrained(model_id)",
                    "        model = AutoModel.from_pretrained(model_id).eval()",
                    "        enc = tok(text, return_tensors=\"pt\", truncation=True)",
                    "        with torch.no_grad():",
                    "            out = model(**enc)",
                    "        h = getattr(out, \"last_hidden_state\", None) or getattr(out, \"pooler_output\", None)",
                    "        if h is None:",
                    "            return torch.zeros(1)",
                    "        return (h.mean(1) if pool == \"mean\" else h[:, 0])[0]",
                    "    except Exception:",
                    "        return torch.zeros(1)",
                ])
                forward_lines.append(f"{dst} = {helper}({_lit(p.get('model_id'))}, {_lit(p.get('text'))})")

            # ---------------- unknown node type ----------------
            else:
                if dst is not None:
                    forward_lines.append(f"{dst} = None  # unsupported node type '{t}'")

        # return: graph_output value(s), else the last computed node output.
        if returns:
            ret = returns[-1] if len(returns) == 1 else "(" + ", ".join(returns) + ")"
        else:
            ret = last_var or "None"

        return init_lines, forward_lines, params, ret


def load_graph_file_from_path(path: Path) -> GraphDocument:
    """Load a .riko file into a GraphDocument (used during recursive inline)."""
    return GraphDocument.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _tail(params: List[str], trainable: bool) -> str:
    """Generate the train / inference demo block."""
    lines: List[str] = []
    if trainable:
        lines += [
            "def train(model, steps=20, lr=1e-3):",
            "    import torch.optim as optim",
            "    optimizer = optim.Adam(model.parameters(), lr=lr)",
            "    losses = []",
            "    for _step in range(steps):",
            "        optimizer.zero_grad()",
            "        loss = model()",
            "        loss.backward()",
            "        optimizer.step()",
            "        losses.append(float(loss.item()))",
            "    return losses",
            "",
            "",
            'if __name__ == "__main__":',
            "    model = GraphModel()",
            "    losses = train(model, steps=20, lr=1e-3)",
            "    for step, loss in enumerate(losses):",
            '        print(f"step {step:3d}: loss = {loss:.4f}")',
        ]
    else:
        lines += [
            'if __name__ == "__main__":',
            "    model = GraphModel()",
            "    model.eval()",
        ]
        if params:
            lines += [
                "    # Provide input via: output = model(<input tensor>)",
                '    print("Model exported. Call model(input) to run.")',
            ]
        else:
            lines += [
                "    output = model()",
                '    print("output:", output)',
            ]
    return "\n".join(lines)


def export_python(doc: GraphDocument, registry: Optional[Registry] = None) -> str:
    """Generate a clean, runnable, self-contained nn.Module Python script."""
    reg = registry if registry is not None else default_registry()
    emitter = _Emitter(reg)
    try:
        init, fwd, params, ret = emitter._generate(doc, "GraphModel")
    except Exception as exc:
        return f"# ERROR: {type(exc).__name__}: {exc}\n"

    has_loss = any(n.type_name in _LOSS_TYPES for n in doc.nodes.values())
    trainable = has_loss and not params

    header = ('"""Generated by Entropia Riko."""\n\n'
              "import torch\nimport torch.nn as nn\n"
              "import torch.nn.functional as F\n")

    blocks: List[str] = []
    if emitter.helpers:
        blocks.append("\n\n".join(emitter.helpers))
    if emitter.nested_classes:
        blocks.extend(emitter.nested_classes)
    blocks.append(emitter._render("GraphModel", init, fwd, params, ret))

    body = "\n\n\n".join(blocks)
    tail = _tail(params, trainable)
    return header + "\n\n" + body + "\n\n\n" + tail + "\n"


def export_python_project(doc: GraphDocument, registry: Optional[Registry] = None) -> List[Dict[str, str]]:
    """Export a multi-file PyTorch project (GitHub-style layout).

    Returns a list of ``{"path": ..., "content": ...}`` entries::

        README.md
        requirements.txt
        src/__init__.py
        src/<name>.py      (the generated nn.Module script)

    The result mirrors the working folder structure so the project is a
    standalone, IDE-openable PyTorch repo equivalent to the source graph.
    """
    reg = registry if registry is not None else default_registry()
    name = (doc.metadata or {}).get("name") or "graph_project"
    safe = re.sub(r"\W+", "_", str(name)).strip("_") or "graph_project"
    model_code = export_python(doc, reg)

    readme = (
        f"# {name}\n\n"
        "Generated by **Entropia Riko** from a node graph.\n\n"
        "## Structure\n\n"
        "```\n"
        "src/\n"
        f"└── {safe}.py   # torch.nn.Module (layers in __init__, ops in forward)\n"
        "```\n\n"
        "## Run\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        f"python -m src.{safe}\n"
        "```\n"
    )

    return [
        {"path": "README.md", "content": readme},
        {"path": "requirements.txt", "content": "torch\n"},
        {"path": "src/__init__.py", "content": '"""Generated by Entropia Riko."""\n'},
        {"path": f"src/{safe}.py", "content": model_code},
    ]
