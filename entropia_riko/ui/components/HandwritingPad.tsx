/** Handwriting pad — draw a 28×28 digit and send it as a `constant` node
 * (shape [1,1,28,28]) to feed the MNIST example for inference. */
import { useEffect, useRef } from "react";
import { useGraphStore } from "../store/graphStore";

export function HandwritingPad() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const addNode = useGraphStore((s) => s.addNode);
  const updateParam = useGraphStore((s) => s.updateParam);
  const pushToast = useGraphStore((s) => s.pushToast);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.lineWidth = 12;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#000";
    }
  }, []);

  const pos = (e: React.MouseEvent) => {
    const c = canvasRef.current!;
    const r = c.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const down = (e: React.MouseEvent) => {
    drawing.current = true;
    const c = canvasRef.current!;
    const ctx = c.getContext("2d")!;
    const { x, y } = pos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y);
    ctx.stroke();
  };
  const move = (e: React.MouseEvent) => {
    if (!drawing.current) return;
    const c = canvasRef.current!;
    const ctx = c.getContext("2d")!;
    const { x, y } = pos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };
  const up = () => {
    drawing.current = false;
  };

  const clear = () => {
    const c = canvasRef.current!;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, c.width, c.height);
  };

  const send = () => {
    const c = canvasRef.current!;
    const small = document.createElement("canvas");
    small.width = 28;
    small.height = 28;
    const sctx = small.getContext("2d")!;
    sctx.drawImage(c, 0, 0, 28, 28);
    const img = sctx.getImageData(0, 0, 28, 28).data;
    const flat: number[] = [];
    for (let i = 0; i < 28 * 28; i++) {
      const v = 1 - (img[i * 4] + img[i * 4 + 1] + img[i * 4 + 2]) / (3 * 255);
      flat.push(Math.round(v * 1000) / 1000);
    }
    const rows: number[][] = [];
    for (let r = 0; r < 28; r++) rows.push(flat.slice(r * 28, r * 28 + 28));
    const value = [[rows]]; // [1, 1, 28, 28]
    addNode("constant", { x: 260, y: 150 });
    const id = useGraphStore.getState().selectedNodeId;
    if (id) {
      updateParam(id, "value", value);
      pushToast("success", "Digit → constant node (1×28×28)");
    }
  };

  return (
    <div className="handpad">
      <div className="panel-head">
        <span className="panel-head-title">Handwriting Pad</span>
        <span className="spacer" />
        <button className="panel-btn" onClick={clear}>Clear</button>
        <button className="panel-btn" onClick={send}>Send → Node</button>
      </div>
      <div className="handpad-canvas">
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          onMouseDown={down}
          onMouseMove={move}
          onMouseUp={up}
          onMouseLeave={up}
        />
      </div>
      <div className="handpad-hint">
        Draw a digit → Send creates a <code>constant</code> node (1×28×28) to feed the MNIST example.
      </div>
    </div>
  );
}
