/** A draggable floating window (detachable-feel) used by all app dialogs. */
import { useRef, useState } from "react";

export function FloatingWindow({
  title,
  onClose,
  x = 140,
  y = 90,
  width = 520,
  zIndex = 1000,
  children,
}: {
  title: string;
  onClose: () => void;
  x?: number;
  y?: number;
  width?: number;
  zIndex?: number;
  children: React.ReactNode;
}) {
  const [pos, setPos] = useState({ x, y });
  // Cursor offset from the window's top-left, captured when a drag begins.
  const drag = useRef<{ dx: number; dy: number } | null>(null);

  const onMouseDown = (e: React.MouseEvent) => {
    drag.current = { dx: e.clientX - pos.x, dy: e.clientY - pos.y };
    const move = (ev: MouseEvent) => {
      if (!drag.current) return;
      // Preserve the original grab offset so the window doesn't jump under the cursor.
      setPos({ x: ev.clientX - drag.current.dx, y: ev.clientY - drag.current.dy });
    };
    const up = () => {
      drag.current = null;
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    // Listen on document so dragging continues even outside the titlebar.
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  };

  return (
    <div className="float-window" style={{ left: pos.x, top: pos.y, width, zIndex }}>
      <div className="float-titlebar" onMouseDown={onMouseDown}>
        <span className="float-title">{title}</span>
        <button className="float-close" onClick={onClose}>✕</button>
      </div>
      <div className="float-body">{children}</div>
    </div>
  );
}
