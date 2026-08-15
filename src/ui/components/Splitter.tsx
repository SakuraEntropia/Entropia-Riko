/** Draggable panel splitters (vertical between columns, horizontal above the
 * status panel). */
import { useRef } from "react";

function useDrag(onDrag: (dx: number, dy: number) => void, cursor: string) {
  const start = useRef<{ x: number; y: number } | null>(null);

  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    start.current = { x: e.clientX, y: e.clientY };
    const move = (ev: MouseEvent) => {
      if (!start.current) return;
      onDrag(ev.clientX - start.current.x, ev.clientY - start.current.y);
      start.current = { x: ev.clientX, y: ev.clientY };
    };
    const up = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
      document.body.style.cursor = "";
    };
    document.body.style.cursor = cursor;
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  };

  return onMouseDown;
}

export function VSplitter({ onDrag }: { onDrag: (dx: number, total: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const onMouseDown = useDrag((dx) => {
    const total = ref.current?.parentElement?.clientWidth ?? 1;
    onDrag(dx, total);
  }, "col-resize");
  return <div ref={ref} className="splitter-v" onMouseDown={onMouseDown} />;
}

export function HSplitter({ onDrag }: { onDrag: (dy: number, total: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const onMouseDown = useDrag((_dx, dy) => {
    const total = ref.current?.parentElement?.clientHeight ?? 1;
    onDrag(dy, total);
  }, "row-resize");
  return <div ref={ref} className="splitter-h" onMouseDown={onMouseDown} />;
}
