"""Graph runtime: validation, dependency ordering, execution (API.md).

Operates on GraphDocument + Registry. Does not import Houdini or torch.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

from ..core.document import GraphDocument
from .registry import Registry, default_registry


class RuntimeExecutionError(Exception):
    """Runtime error carrying node / port context."""


def validate(doc: GraphDocument, registry: Optional[Registry] = None) -> List[str]:
    """Return a list of validation error strings (empty means valid)."""
    reg = registry if registry is not None else default_registry()
    errors: List[str] = []

    # Node types must be registered.
    for nid, node in doc.nodes.items():
        if node.type_name not in reg:
            errors.append(f"节点 '{nid}' 类型 '{node.type_name}' 未注册")

    # Edges must reference existing nodes and ports.
    for e in doc.edges:
        if e.source_node not in doc.nodes:
            errors.append(f"连接 {e.id}: 源节点 '{e.source_node}' 不存在")
            continue
        if e.target_node not in doc.nodes:
            errors.append(f"连接 {e.id}: 目标节点 '{e.target_node}' 不存在")
            continue
        src = doc.nodes[e.source_node]
        dst = doc.nodes[e.target_node]
        if src.type_name in reg:
            src_cls = reg.get(src.type_name)
            if e.source_port not in [o.name for o in src_cls.outputs]:
                errors.append(
                    f"连接 {e.id}: 节点 '{e.source_node}'({src.type_name})"
                    f"无输出端口 '{e.source_port}'"
                )
        if dst.type_name in reg:
            dst_cls = reg.get(dst.type_name)
            if e.target_port not in [i.name for i in dst_cls.inputs]:
                errors.append(
                    f"连接 {e.id}: 节点 '{e.target_node}'({dst.type_name})"
                    f"无输入端口 '{e.target_port}'"
                )

    # Required inputs must be connected.
    connected = {(e.target_node, e.target_port) for e in doc.edges}
    for nid, node in doc.nodes.items():
        if node.type_name not in reg:
            continue
        cls = reg.get(node.type_name)
        for inp in cls.inputs:
            if inp.required and (nid, inp.name) not in connected:
                errors.append(
                    f"节点 '{nid}'({node.type_name}) 缺少必需输入 '{inp.name}'"
                )

    # Cycle detection (only if no earlier errors).
    if not errors:
        try:
            execution_order(doc)
        except RuntimeExecutionError as exc:
            errors.append(str(exc))

    return errors


def execution_order(doc: GraphDocument) -> List[str]:
    """Kahn topological sort; raise RuntimeExecutionError on cycles."""
    indeg = {nid: 0 for nid in doc.nodes}
    adj: Dict[str, List[str]] = {nid: [] for nid in doc.nodes}
    for e in doc.edges:
        adj[e.source_node].append(e.target_node)
        indeg[e.target_node] += 1

    q = deque(nid for nid in doc.nodes if indeg[nid] == 0)
    order: List[str] = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)

    if len(order) != len(doc.nodes):
        remaining = [n for n in doc.nodes if n not in set(order)]
        raise RuntimeExecutionError(f"图中存在环，涉及节点: {remaining}")
    return order


def execute(
    doc: GraphDocument,
    registry: Optional[Registry] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Validate, order, and execute the graph; return {node_id: {port: value}}."""
    reg = registry if registry is not None else default_registry()
    errors = validate(doc, reg)
    if errors:
        raise RuntimeExecutionError(
            "图校验失败:\n  - " + "\n  - ".join(errors)
        )
    order = execution_order(doc)
    ctx = context or {}
    outputs: Dict[str, Dict[str, Any]] = {}

    in_edges = {
        (e.target_node, e.target_port): (e.source_node, e.source_port)
        for e in doc.edges
    }

    for nid in order:
        node = doc.nodes[nid]
        cls = reg.get(node.type_name)
        instance = cls(node.parameters)
        inputs: Dict[str, Any] = {}
        for inp in cls.inputs:
            key = (nid, inp.name)
            if key in in_edges:
                sid, sport = in_edges[key]
                inputs[inp.name] = outputs[sid][sport]
            elif inp.default is not None:
                inputs[inp.name] = inp.default

        instance.validate_inputs(inputs)
        try:
            out = instance.execute(inputs, instance.params, ctx)
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeExecutionError(
                f"节点 '{nid}'({node.type_name}) 执行失败: {exc}"
            ) from exc
        if not isinstance(out, dict):
            raise RuntimeExecutionError(
                f"节点 '{nid}'({node.type_name}) execute 必须返回 dict，"
                f"实际 {type(out).__name__}"
            )
        outputs[nid] = out

    return outputs
