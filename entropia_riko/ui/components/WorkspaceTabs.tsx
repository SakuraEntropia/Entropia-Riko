/** Blender-style workspace tabs + "+" menu + right-click context menu (unified PopupMenu). */
import { useEffect, useMemo, useRef, useState } from "react";
import { PopupMenu, type MenuEntry } from "./PopupMenu";
import {
  WORKSPACE_PRESETS,
  type WorkspaceInstance,
} from "../areas";

interface CtxState {
  id: string;
  x: number;
  y: number;
}

export function WorkspaceTabs({
  workspaces,
  activeId,
  onSwitch,
  onAdd,
  onRemove,
  onRename,
  onDuplicate,
  onMove,
}: {
  workspaces: WorkspaceInstance[];
  activeId: string;
  onSwitch: (id: string) => void;
  onAdd: (presetId: string) => void;
  onRemove: (id: string) => void;
  onRename: (id: string, name: string) => void;
  onDuplicate: (id: string) => void;
  onMove: (id: string, delta: number) => void;
}) {
  const [addOpen, setAddOpen] = useState(false);
  const [addPos, setAddPos] = useState<{ x: number; y: number } | null>(null);
  const [ctx, setCtx] = useState<CtxState | null>(null);
  const [overflowing, setOverflowing] = useState(false);
  const tabsRef = useRef<HTMLDivElement>(null);

  // Detect horizontal overflow so CSS can apply the fade mask at the clipped edge.
  useEffect(() => {
    const el = tabsRef.current;
    if (!el) return;
    const check = () => setOverflowing(el.scrollWidth > el.clientWidth + 1);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, [workspaces]);

  const canRemove = workspaces.length > 1;

  // One menu entry per built-in workspace preset, shown by the "+" button.
  const presetEntries = useMemo<MenuEntry[]>(
    () =>
      WORKSPACE_PRESETS.map((p) => ({
        id: p.id,
        label: p.label,
        description: p.description,
        category: p.category,
        onSelect: () => onAdd(p.id),
      })),
    [onAdd]
  );

  // Anchor the preset menu just below the "+" button's left edge.
  const openAdd = (e: React.MouseEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setAddPos({ x: r.left, y: r.bottom + 6 });
    setAddOpen(true);
  };

  const rename = (id: string) => {
    const ws = workspaces.find((w) => w.id === id);
    const name = window.prompt("Rename workspace", ws?.name ?? "");
    if (name && name.trim()) onRename(id, name.trim());
  };

  // Per-tab right-click actions; "Close" is hidden when only one tab remains.
  const ctxEntries = (id: string): MenuEntry[] => [
    { id: "rename", label: "Rename", onSelect: () => rename(id) },
    { id: "duplicate", label: "Duplicate", onSelect: () => onDuplicate(id) },
    { id: "sep1", label: "", separator: true },
    { id: "left", label: "← Move Left", onSelect: () => onMove(id, -1) },
    { id: "right", label: "Move Right →", onSelect: () => onMove(id, 1) },
    { id: "sep2", label: "", separator: true },
    ...(canRemove ? [{ id: "close", label: "Close", danger: true, onSelect: () => onRemove(id) }] : []),
  ];

  return (
    <div className="workspace-tabs-wrap">
      <div ref={tabsRef} className={`workspace-tabs ${overflowing ? "overflowing" : ""}`}>
        {workspaces.map((ws) => (
          <div
            key={ws.id}
            className={`workspace-tab ${ws.id === activeId ? "active" : ""}`}
            onClick={() => onSwitch(ws.id)}
            onContextMenu={(e) => {
              e.preventDefault();
              setCtx({ id: ws.id, x: e.clientX, y: e.clientY });
            }}
          >
            <span>{ws.name}</span>
            {canRemove && (
              <span
                className="workspace-tab-close"
                title="Close workspace"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove(ws.id);
                }}
              >
                ✕
              </span>
            )}
          </div>
        ))}
      </div>
      <button className="workspace-add" title="Add workspace" onClick={openAdd}>
        +
      </button>

      {addOpen && addPos && (
        <PopupMenu entries={presetEntries} x={addPos.x} y={addPos.y} onClose={() => setAddOpen(false)} />
      )}

      {ctx && (
        <PopupMenu
          entries={ctxEntries(ctx.id)}
          x={ctx.x}
          y={ctx.y}
          onClose={() => setCtx(null)}
        />
      )}
    </div>
  );
}
