/** Subgraph authoring: track multi-selection and collapse a selection into a
 * `.riko` subgraph referenced by an `import` node, preserving boundary edges by
 * inserting `graph_input` / `graph_output` nodes at the subgraph boundary. */
import type { Node, Edge } from "@xyflow/react";
import { useGraphStore, type ThnNodeData } from "./store/graphStore";

// Selected node ids live here (module-level) so the canvas selection and the
// context-menu action stay decoupled from the main graph store shape.
let selectedNodeIds: string[] = [];

export function setSelectedNodeIds(ids: string[]): void {
  selectedNodeIds = ids;
}

export function getSelectedNodeIds(): string[] {
  return selectedNodeIds;
}

let _edgeCounter = 0;
function edgeId(prefix: string): string {
  _edgeCounter += 1;
  return `${prefix}_${Date.now()}_${_edgeCounter}`;
}

export async function createSubgraphFromSelection(): Promise<void> {
  const s = useGraphStore.getState();
  const ids = selectedNodeIds.filter((id) => s.nodes.some((n) => n.id === id));
  if (ids.length === 0) {
    s.pushToast("error", "先框选一些节点再创建子图");
    return;
  }
  const name = window.prompt("Subgraph name", "subgraph");
  if (!name || !name.trim()) return;

  const sel = new Set(ids);
  const subNodes = s.nodes.filter((n) => sel.has(n.id));
  const subEdges = s.edges.filter((e) => sel.has(e.source) && sel.has(e.target));

  // Boundary edges: external → internal (inputs) and internal → external (outputs).
  const incoming = s.edges.filter((e) => !sel.has(e.source) && sel.has(e.target));
  const outgoing = s.edges.filter((e) => sel.has(e.source) && !sel.has(e.target));

  // Insert graph_input / graph_output nodes at the boundary so the subgraph keeps
  // its own well-defined interface. The first input/output is named "input" /
  // "output" to match the subgraph execution convention.
  const extraNodes: Record<string, unknown>[] = [];
  const extraEdges: Record<string, unknown>[] = [];
  // Named-port convention: input / input_2 / input_3 ↔ graph_input(name=...),
  // output / output_2 / output_3 ↔ graph_output(name=...).
  const inPort = (i: number) => (i === 0 ? "input" : `input_${i + 1}`);
  const outPort = (i: number) => (i === 0 ? "output" : `output_${i + 1}`);

  incoming.forEach((e, i) => {
    const portName = inPort(i);
    const gid = `gin_${i}`;
    extraNodes.push({
      id: gid,
      type_name: "graph_input",
      label: `Input ${i}`,
      category: "Subgraph",
      position: [0, i * 90],
      parameters: { name: portName, data_kind: "tensor" },
      inputs: [],
      outputs: [],
    });
    extraEdges.push({
      id: `be_in_${i}`,
      source_node: gid,
      source_port: "value",
      target_node: e.target,
      target_port: e.targetHandle,
    });
  });
  outgoing.forEach((e, i) => {
    const portName = outPort(i);
    const gid = `gout_${i}`;
    extraNodes.push({
      id: gid,
      type_name: "graph_output",
      label: `Output ${i}`,
      category: "Subgraph",
      position: [1200, i * 90],
      parameters: { name: portName, data_kind: "tensor" },
      inputs: [],
      outputs: [],
    });
    extraEdges.push({
      id: `be_out_${i}`,
      source_node: e.source,
      source_port: e.sourceHandle,
      target_node: gid,
      target_port: "value",
    });
  });

  // Build the subgraph document (same shape as a saved .riko).
  const doc = {
    version: "1.0",
    metadata: { name: name.trim(), description: "" },
    nodes: [
      ...subNodes.map((n) => ({
        id: n.id,
        type_name: n.data.type,
        label: n.data.label,
        category: n.data.category,
        position: [n.position.x, n.position.y],
        parameters: n.data.params,
        inputs: [],
        outputs: [],
      })),
      ...extraNodes,
    ],
    edges: [
      ...subEdges.map((e) => ({
        id: e.id,
        source_node: e.source,
        source_port: e.sourceHandle,
        target_node: e.target,
        target_port: e.targetHandle,
      })),
      ...extraEdges,
    ],
    settings: {},
  };

  try {
    const resp = await fetch("/api/files/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name.trim(), format: "ascii", doc }),
    });
    const data = await resp.json();
    if (data.status !== "success") {
      s.pushToast("error", data.error ?? "子图保存失败");
      return;
    }

    // Replace the selection with a single `import` node at its centroid.
    const cx = subNodes.reduce((sum, n) => sum + n.position.x, 0) / subNodes.length;
    const cy = subNodes.reduce((sum, n) => sum + n.position.y, 0) / subNodes.length;
    const def =
      s.nodeDefs.find((d) => d.type === "import") ??
      s.nodeDefs.find((d) => d.type === "graph_reference");
    const importNode: Node<ThnNodeData> = {
      id: `imp_${Date.now()}`,
      type: "thnNode",
      position: { x: cx, y: cy },
      data: {
        type: def?.type ?? "import",
        label: name.trim(),
        category: "Subgraph",
        params:
          def?.type === "import"
            ? { module: name.trim() }
            : { file: `${name.trim()}.riko` },
        inputs: def?.inputs ?? [],
        outputs: def?.outputs ?? [],
        hasError: false,
      },
    };

    // Keep non-boundary edges, and rewire boundary edges through the import node's
    // named multi-ports (up to 3 in / 3 out).
    const remainingEdges: Edge[] = s.edges.filter(
      (e) => !sel.has(e.source) && !sel.has(e.target)
    );
    const MAX_PORTS = 3;
    incoming.slice(0, MAX_PORTS).forEach((e, i) => {
      remainingEdges.push({
        id: edgeId("rwin"),
        source: e.source,
        target: importNode.id,
        sourceHandle: e.sourceHandle,
        targetHandle: inPort(i),
      });
    });
    outgoing.slice(0, MAX_PORTS).forEach((e, i) => {
      remainingEdges.push({
        id: edgeId("rwout"),
        source: importNode.id,
        target: e.target,
        sourceHandle: outPort(i),
        targetHandle: e.targetHandle,
      });
    });

    useGraphStore.setState({
      nodes: [...s.nodes.filter((n) => !sel.has(n.id)), importNode],
      edges: remainingEdges,
      selectedNodeId: null,
      dirty: true,
    });
    selectedNodeIds = [];
    const droppedIn = Math.max(0, incoming.length - MAX_PORTS);
    const droppedOut = Math.max(0, outgoing.length - MAX_PORTS);
    if (droppedIn || droppedOut) {
      s.pushToast(
        "info",
        `子图已创建，但超过 3 入/3 出的边界连线有 ${droppedIn} 入/${droppedOut} 出 未保留`
      );
    } else {
      s.pushToast("success", `已创建子图 ${name.trim()}（边界连线已保留）`);
    }
    s.refreshFiles();
  } catch (e) {
    s.pushToast("error", e instanceof Error ? e.message : String(e));
  }
}
