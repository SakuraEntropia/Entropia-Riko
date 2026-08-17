/** Center graph canvas using React Flow, with a pill zoom control docked right,
 * and node frames (visual grouping boxes) rendered inside the viewport. */
import { useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  useReactFlow,
  ViewportPortal,
  type NodeTypes,
} from "@xyflow/react";
import { useGraphStore, type Frame } from "../store/graphStore";
import { setSelectedNodeIds } from "../subgraphActions";
import { NodeCard } from "./NodeCard";
import { ContextMenu, type ContextMenuState } from "./ContextMenu";

// Map the custom "thnNode" type id to its card component.
const nodeTypes: NodeTypes = { thnNode: NodeCard };

function FlowCanvas() {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const frames = useGraphStore((s) => s.frames);
  const updateFrame = useGraphStore((s) => s.updateFrame);
  const removeFrame = useGraphStore((s) => s.removeFrame);
  const onNodesChange = useGraphStore((s) => s.onNodesChange);
  const onEdgesChange = useGraphStore((s) => s.onEdgesChange);
  const onConnect = useGraphStore((s) => s.onConnect);
  const onSelectNode = useGraphStore((s) => s.onSelectNode);
  const enterSubgraph = useGraphStore((s) => s.enterSubgraph);
  const { zoomIn, zoomOut, fitView, getViewport } = useReactFlow();

  const onFrameDrag = (e: React.MouseEvent, frame: Frame) => {
    // Ignore drags that begin on the frame's close button.
    if ((e.target as HTMLElement).closest("button")) return;
    e.stopPropagation();
    const startX = e.clientX;
    const startY = e.clientY;
    const origX = frame.x;
    const origY = frame.y;
    const move = (ev: MouseEvent) => {
      const zoom = getViewport().zoom || 1;
      // Divide by zoom so a screen-pixel drag maps to graph-world units.
      updateFrame(frame.id, {
        x: origX + (ev.clientX - startX) / zoom,
        y: origY + (ev.clientY - startY) / zoom,
      });
    };
    const up = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  };

  return (
    <>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={(_evt, node) => onSelectNode(node.id)}
        onNodeDoubleClick={(_evt, node) => {
          // Double-clicking a reference/import node drills into its subgraph.
          if (node.data.type === "graph_reference" || node.data.type === "import") {
            enterSubgraph(node.id);
          }
        }}
        onPaneClick={() => onSelectNode(null)}
        onSelectionChange={({ nodes }) => setSelectedNodeIds(nodes.map((n) => n.id))}
        selectionOnDrag
        panOnDrag={[2]}
        fitView
      >
        <Background />
        {/* Frames render inside the viewport portal so they pan and zoom with the graph. */}
        <ViewportPortal>
          {frames.map((f) => (
            <div
              key={f.id}
              className="graph-frame"
              style={{ left: f.x, top: f.y, width: f.width, height: f.height }}
              onMouseDown={(e) => onFrameDrag(e, f)}
              onDoubleClick={() => {
                const t = window.prompt("Frame title", f.title);
                if (t !== null) updateFrame(f.id, { title: t });
              }}
            >
              <div className="graph-frame-title">
                {f.title}
                <button className="graph-frame-close" onClick={() => removeFrame(f.id)}>
                  ✕
                </button>
              </div>
            </div>
          ))}
        </ViewportPortal>
      </ReactFlow>
      {/* Pill zoom control docked at the right: zoom in, fit view, zoom out. */}
      <div className="graph-zoom">
        <button onClick={() => zoomIn()} title="Zoom in">+</button>
        <button onClick={() => fitView()} title="Fit view">⤢</button>
        <button onClick={() => zoomOut()} title="Zoom out">−</button>
      </div>
    </>
  );
}

export function GraphCanvas() {
  const [ctxMenu, setCtxMenu] = useState<ContextMenuState>({ x: 0, y: 0, visible: false });
  const graphStack = useGraphStore((s) => s.graphStack);
  const exitToLevel = useGraphStore((s) => s.exitToLevel);

  return (
    <div
      className="graph-canvas"
      onContextMenu={(e) => {
        e.preventDefault();
        setCtxMenu({ x: e.clientX, y: e.clientY, visible: true });
      }}
    >
      <ReactFlowProvider>
        <FlowCanvas />
      </ReactFlowProvider>
      {/* Subgraph breadcrumb: click an ancestor to jump back up the graph stack. */}
      <div className="graph-breadcrumb">
        {graphStack.map((l, i) => (
          <span key={i} className="graph-breadcrumb-wrap">
            {i > 0 && <span className="graph-breadcrumb-sep">/</span>}
            <span
              className={`graph-breadcrumb-crumb ${i === graphStack.length - 1 ? "current" : ""}`}
              onClick={() => {
                if (i < graphStack.length - 1) exitToLevel(i);
              }}
            >
              {l.name || "root"}
            </span>
          </span>
        ))}
      </div>
      <ContextMenu state={ctxMenu} onClose={() => setCtxMenu({ x: 0, y: 0, visible: false })} />
    </div>
  );
}
