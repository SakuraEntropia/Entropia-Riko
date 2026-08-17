/** Modular panel system (Blender-style editor areas).

Each panel has a slim header bar (type dropdown + close) and a bottom-right
corner grip:

- drag up/left  → split (a live preview line follows the mouse, clipped to the
  panel area),
- drag down/right → merge (the sibling panel is shaded + bold-bordered),
- release to commit.
*/
import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { NodeLibrary } from "./NodeLibrary";
import { FileManager } from "./FileManager";
import { GraphCanvas } from "./GraphCanvas";
import { SideInspector } from "./SideInspector";
import { StatusPanel } from "./StatusPanel";
import { LossPanel } from "./LossPanel";
import { DocsPanel } from "./DocsPanel";
import { PluginPanel } from "./PluginPanel";
import { ProjectPanel } from "./ProjectPanel";
import { CodeEditor } from "./CodeEditor";
import { HandwritingPad } from "./HandwritingPad";

export type PanelType =
  | "nodes"
  | "files"
  | "project"
  | "canvas"
  | "inspector"
  | "status"
  | "loss"
  | "docs"
  | "plugins"
  | "code"
  | "pad";

export interface PanelSpec {
  id: string;
  type: PanelType;
  /** Fixed size in px; 0 means "flexible" (fills remaining space). */
  size: number;
}

export type DragMode = "split-row" | "split-column" | "merge";

export interface DragPreview {
  mode: DragMode;
  /** Position relative to the panel being dragged (clips the preview line). */
  x: number;
  y: number;
}

export const PANEL_TYPES: { value: PanelType; label: string; category: string }[] = [
  { value: "canvas", label: "Graph", category: "Editor" },
  { value: "inspector", label: "Inspector", category: "Editor" },
  { value: "status", label: "Status / Logs", category: "Editor" },
  { value: "loss", label: "Loss Curve", category: "Editor" },
  { value: "nodes", label: "Node Library", category: "Data" },
  { value: "files", label: "Asset Library", category: "Data" },
  { value: "project", label: "New File", category: "Data" },
  { value: "code", label: "Code Editor", category: "Tools" },
  { value: "pad", label: "Handwriting Pad", category: "Tools" },
  { value: "docs", label: "Documentation", category: "Tools" },
  { value: "plugins", label: "Plugins", category: "Tools" },
];

const PANEL_CATEGORIES = ["Editor", "Data", "Tools"];

export function panelLabel(type: PanelType): string {
  return PANEL_TYPES.find((t) => t.value === type)?.label ?? type;
}

// --- pluggable content registry -------------------------------------------------
// The area-tree panel system is decoupled from the app: it renders whatever the
// registry maps a window type to. Other apps can reuse the panel/layout system
// and call `registerPanelContent(...)` with their own components.
type PanelRenderer = () => React.ReactNode;

const contentRegistry: Record<string, PanelRenderer> = {
  nodes: () => <NodeLibrary />,
  files: () => <FileManager />,
  project: () => <ProjectPanel />,
  canvas: () => <GraphCanvas />,
  inspector: () => <SideInspector />,
  status: () => <StatusPanel />,
  loss: () => <LossPanel />,
  docs: () => <DocsPanel />,
  plugins: () => <PluginPanel />,
  code: () => <CodeEditor />,
  pad: () => <HandwritingPad />,
};

export function registerPanelContent(type: string, render: PanelRenderer): void {
  contentRegistry[type] = render;
}

export function renderPanelContent(type: PanelType): React.ReactNode {
  const render = contentRegistry[type];
  return render ? render() : null;
}

function TypeDropdown({
  type,
  onType,
}: {
  type: PanelType;
  onType: (t: PanelType) => void;
}) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const btnRef = useRef<HTMLButtonElement>(null);

  const toggle = () => {
    if (!open && btnRef.current) {
      const r = btnRef.current.getBoundingClientRect();
      // Anchor the menu below the button, right-aligned to its fixed 300px width.
      setPos({ top: r.bottom + 4, left: r.right - 300 });
    }
    setOpen((o) => !o);
  };

  return (
    <div className="type-dropdown">
      <button ref={btnRef} className="panel-btn" title="Switch window type" onClick={toggle}>
        {panelLabel(type)} ▾
      </button>
      {open &&
        createPortal(
          <>
            <div className="type-menu-overlay" onClick={() => setOpen(false)} />
            <div className="type-menu" style={pos ? { position: "fixed", top: pos.top, left: pos.left } : undefined}>
              {PANEL_CATEGORIES.map((cat) => (
                <div key={cat} className="type-menu-col">
                  <div className="type-menu-cat">{cat}</div>
                  {PANEL_TYPES.filter((t) => t.category === cat).map((t) => (
                    <div
                      key={t.value}
                      className={`type-menu-item ${t.value === type ? "active" : ""}`}
                      onClick={() => {
                        onType(t.value);
                        setOpen(false);
                      }}
                    >
                      {t.label}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </>,
          document.body
        )}
    </div>
  );
}

function computeMode(dx: number, dy: number): DragMode | null {
  // Ignore tiny movements so an accidental click doesn't trigger a split/merge.
  if (Math.abs(dx) < 14 && Math.abs(dy) < 14) return null;
  // Horizontal drags win: left → split into a column, right → merge.
  if (Math.abs(dx) > Math.abs(dy)) return dx < 0 ? "split-row" : "merge";
  // Vertical drags: up → split into a row, down → merge.
  return dy < 0 ? "split-column" : "merge";
}

function CornerGrip({
  onPreview,
  onCommit,
}: {
  onPreview: (p: DragPreview | null) => void;
  onCommit: (mode: DragMode) => void;
}) {
  const start = useRef<{ x: number; y: number } | null>(null);

  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Capture the panel slot's bounds so preview coords stay clipped to this panel.
    const slot = (e.currentTarget as HTMLElement).parentElement;
    const rect = slot?.getBoundingClientRect() ?? null;
    start.current = { x: e.clientX, y: e.clientY };

    const move = (ev: MouseEvent) => {
      if (!start.current) return;
      const dx = ev.clientX - start.current.x;
      const dy = ev.clientY - start.current.y;
      const mode = computeMode(dx, dy);
      // Report the cursor offset from the panel's top-left for the preview overlay.
      onPreview(
        mode
          ? { mode, x: ev.clientX - (rect?.left ?? 0), y: ev.clientY - (rect?.top ?? 0) }
          : null
      );
    };
    const up = (ev: MouseEvent) => {
      if (!start.current) return;
      const dx = ev.clientX - start.current.x;
      const dy = ev.clientY - start.current.y;
      // Tear down listeners and restore the cursor before committing the action.
      start.current = null;
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
      document.body.style.cursor = "";
      const mode = computeMode(dx, dy);
      onPreview(null);
      if (mode) onCommit(mode);
    };

    document.body.style.cursor = "crosshair";
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  };

  return (
    <div
      className="corner-grip"
      title="拖左上：分离面板 · 拖右下：合并面板"
      onMouseDown={onMouseDown}
    />
  );
}

export function PanelSlot({
  id,
  type,
  onType,
  onSplitRow,
  onSplitColumn,
  onMerge,
  onClose,
  canMerge,
  mergeTarget,
  preview,
  onPreview,
}: {
  id: string;
  type: PanelType;
  onType: (t: PanelType) => void;
  onSplitRow: () => void;
  onSplitColumn: () => void;
  onMerge: () => void;
  onClose: () => void;
  canMerge: boolean;
  mergeTarget: boolean;
  preview: DragPreview | null;
  onPreview: (p: DragPreview | null) => void;
}) {
  return (
    <div className={`panel-slot ${mergeTarget ? "merge-target" : ""}`}>
      <div className="panel-bar">
        <div className="panel-bar-spacer" />
        <TypeDropdown type={type} onType={onType} />
        {canMerge && (
          <button className="panel-btn" title="Close / merge panel away" onClick={onClose}>
            ✕
          </button>
        )}
      </div>
      <div className="panel-body">{renderPanelContent(type)}</div>
      {/* Row-split preview: two shaded bands flanking the vertical split line. */}
      {preview && preview.mode === "split-row" && (
        <>
          <div className="split-preview" style={{ left: 0, top: 0, bottom: 0, width: Math.max(0, preview.x - 4) }} />
          <div className="split-preview" style={{ left: preview.x + 4, top: 0, bottom: 0, right: 0 }} />
        </>
      )}
      {/* Column-split preview: two shaded bands flanking the horizontal split line. */}
      {preview && preview.mode === "split-column" && (
        <>
          <div className="split-preview" style={{ left: 0, right: 0, top: 0, height: Math.max(0, preview.y - 4) }} />
          <div className="split-preview" style={{ left: 0, right: 0, top: preview.y + 4, bottom: 0 }} />
        </>
      )}
      <CornerGrip
        onPreview={onPreview}
        onCommit={(mode) =>
          mode === "split-row" ? onSplitRow() : mode === "split-column" ? onSplitColumn() : onMerge()
        }
      />
    </div>
  );
}
