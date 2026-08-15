"""Demo: complete transformer pipeline as a node graph.

Pipeline:
  constant(tokens) -> embedding -> transformer_encoder -> linear -> output

  tokens (2,5) int -> embedding (2,5,16) -> transformer (2,5,16) -> linear (2,5,5)

Run:
  cd entropia-riko
  .venv/bin/python examples/transformer_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.nodes  # noqa: F401  触发全部节点注册
from src.core.document import GraphDocument, NodeModel, EdgeModel
from src.runtime.executor import execute


def main():
    # Token IDs: batch=2, seq=5 (values in [0, 10))
    tokens = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]

    doc = GraphDocument()
    doc.add_node(NodeModel(
        id="tokens", type_name="constant",
        parameters={"value": tokens},
    ))
    doc.add_node(NodeModel(
        id="emb", type_name="embedding",
        parameters={"num_embeddings": 10, "embedding_dim": 16},
    ))
    doc.add_node(NodeModel(
        id="enc", type_name="transformer_encoder",
        parameters={"d_model": 16, "nhead": 4, "num_layers": 2},
    ))
    doc.add_node(NodeModel(
        id="out", type_name="linear",
        parameters={"in_features": 16, "out_features": 5},
    ))

    doc.add_edge(EdgeModel(
        id="e1", source_node="tokens", source_port="value",
        target_node="emb", target_port="indices",
    ))
    doc.add_edge(EdgeModel(
        id="e2", source_node="emb", source_port="result",
        target_node="enc", target_port="x",
    ))
    doc.add_edge(EdgeModel(
        id="e3", source_node="enc", source_port="output",
        target_node="out", target_port="x",
    ))

    print("=== Transformer Pipeline Demo ===")
    print(f"Input: tokens {tokens} -> shape (2, 5)")
    print()

    outputs = execute(doc)

    for nid in ["emb", "enc", "out"]:
        node = doc.nodes[nid]
        tv = outputs[nid]
        port = next(iter(tv))
        print(f"  {node.label} ({nid}):")
        print(f"    {tv[port].summary()}")
        print()

    print("Pipeline: tokens(2,5) -> embedding(2,5,16) -> transformer(2,5,16) -> linear(2,5,5)")
    print("Done.")


if __name__ == "__main__":
    main()
