"""Graph document model (API.md, APP_ARCHITECTURE.md, DATA_FORMAT.md).

UI graph state: nodes (with positions), edges, ports, settings.
Serializable to dict / JSON so workflows can be saved and loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class PortModel:
    """A node input or output port."""

    name: str
    label: str
    data_kind: str = "tensor"
    direction: str = "in"  # "in" | "out"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "data_kind": self.data_kind,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PortModel:
        return cls(
            name=d["name"],
            label=d.get("label", d["name"]),
            data_kind=d.get("data_kind", "tensor"),
            direction=d.get("direction", "in"),
        )


@dataclass
class NodeModel:
    """A node instance in the graph document."""

    id: str
    type_name: str
    label: str = ""
    category: str = ""
    position: Tuple[float, float] = (0.0, 0.0)
    parameters: Dict[str, Any] = field(default_factory=dict)
    inputs: List[PortModel] = field(default_factory=list)
    outputs: List[PortModel] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type_name": self.type_name,
            "label": self.label,
            "category": self.category,
            "position": list(self.position),
            "parameters": dict(self.parameters),
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NodeModel:
        return cls(
            id=d["id"],
            type_name=d["type_name"],
            label=d.get("label", ""),
            category=d.get("category", ""),
            position=tuple(d.get("position", [0.0, 0.0])),
            parameters=dict(d.get("parameters", {})),
            inputs=[PortModel.from_dict(p) for p in d.get("inputs", [])],
            outputs=[PortModel.from_dict(p) for p in d.get("outputs", [])],
        )


@dataclass
class EdgeModel:
    """A connection from a source port to a target port."""

    id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EdgeModel:
        return cls(
            id=d["id"],
            source_node=d["source_node"],
            source_port=d["source_port"],
            target_node=d["target_node"],
            target_port=d["target_port"],
        )


@dataclass
class GraphDocument:
    """A saveable graph workflow (DATA_FORMAT.md graph document)."""

    version: str = "1.0"
    nodes: Dict[str, NodeModel] = field(default_factory=dict)
    edges: List[EdgeModel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: NodeModel) -> None:
        if node.id in self.nodes:
            raise ValueError(
                f"what: 节点 id '{node.id}' 已存在。\n"
                f"where: core.document.GraphDocument.add_node"
            )
        self.nodes[node.id] = node

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = [
            e for e in self.edges
            if e.source_node != node_id and e.target_node != node_id
        ]

    def add_edge(self, edge: EdgeModel) -> None:
        if edge.source_node not in self.nodes:
            raise ValueError(f"源节点 '{edge.source_node}' 不存在")
        if edge.target_node not in self.nodes:
            raise ValueError(f"目标节点 '{edge.target_node}' 不存在")
        self.edges.append(edge)

    def remove_edge(self, edge_id: str) -> None:
        self.edges = [e for e in self.edges if e.id != edge_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "metadata": dict(self.metadata),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "settings": dict(self.settings),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GraphDocument:
        doc = cls(version=d.get("version", "1.0"))
        doc.metadata = dict(d.get("metadata", {}))
        for n in d.get("nodes", []):
            doc.add_node(NodeModel.from_dict(n))
        for e in d.get("edges", []):
            doc.add_edge(EdgeModel.from_dict(e))
        doc.settings = dict(d.get("settings", {}))
        return doc

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    # ------------------------------------------------------------------ binary
    def to_binary(self) -> bytes:
        """Serialize to the binary ``.ric`` format (magic + version + zlib JSON)."""
        import json
        import zlib

        payload = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return _RIC_MAGIC + _RIC_VERSION + zlib.compress(payload)

    @classmethod
    def from_binary(cls, data: bytes) -> GraphDocument:
        """Deserialize a binary ``.ric`` file back to a GraphDocument."""
        import json
        import zlib

        if not isinstance(data, (bytes, bytearray)) or not data.startswith(_RIC_MAGIC):
            raise ValueError(
                "what: 不是有效的 .ric 二进制文件（缺少 magic 头）。\n"
                "where: core.document.GraphDocument.from_binary"
            )
        payload = zlib.decompress(bytes(data)[len(_RIC_MAGIC) + len(_RIC_VERSION):])
        return cls.from_dict(json.loads(payload.decode("utf-8")))


# Magic + version header for the binary `.ric` format.
_RIC_MAGIC = b"ERIK"
_RIC_VERSION = b"\x01"
