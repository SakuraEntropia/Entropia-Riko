/** Right-click context menu for the graph canvas (unified PopupMenu). */
import { useMemo } from "react";
import { useGraphStore } from "../store/graphStore";
import { createSubgraphFromSelection, getSelectedNodeIds } from "../subgraphActions";
import { PopupMenu, type MenuEntry } from "./PopupMenu";

export interface ContextMenuState {
  x: number;
  y: number;
  visible: boolean;
}

export function ContextMenu({ state, onClose }: { state: ContextMenuState; onClose: () => void }) {
  const nodeDefs = useGraphStore((s) => s.nodeDefs);
  const addNode = useGraphStore((s) => s.addNode);
  const addFrame = useGraphStore((s) => s.addFrame);

  // Fixed "Add Frame" entry plus one entry per registered node type. The click
  // point is offset so the new item is dropped roughly centered under the cursor.
  const entries = useMemo<MenuEntry[]>(
    () => [
      {
        id: "create_subgraph",
        label: `⤵ Create Subgraph (${getSelectedNodeIds().length} selected)`,
        description: "collapse selection into an import node",
        category: "Subgraph",
        disabled: getSelectedNodeIds().length === 0,
        onSelect: () => createSubgraphFromSelection(),
      },
      {
        id: "add_frame",
        label: "＋ Add Frame",
        description: "visual grouping box",
        category: "Frame",
        onSelect: () => addFrame("Frame", { x: state.x - 200, y: state.y - 150 }),
      },
      ...nodeDefs.map((def) => ({
        id: def.type,
        label: def.label,
        description: def.type,
        category: def.category || "Other",
        onSelect: () => addNode(def.type, { x: state.x - 200, y: state.y - 150 }),
      })),
    ],
    [nodeDefs, addNode, addFrame, state.x, state.y]
  );

  if (!state.visible) return null;

  return <PopupMenu entries={entries} x={state.x} y={state.y} onClose={onClose} searchable />;
}
