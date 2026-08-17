/** Loss curve panel: arbitrary training steps/lr + a live SVG loss chart. */
import { useState } from "react";
import { useGraphStore } from "../store/graphStore";

export function LossPanel() {
  const losses = useGraphStore((s) => s.losses);
  const train = useGraphStore((s) => s.train);
  const status = useGraphStore((s) => s.status);

  const [steps, setSteps] = useState(20);
  const [lr, setLr] = useState(1e-3);

  const w = 300;
  const h = 160;
  const pad = 10;

  const points = (() => {
    if (losses.length === 0) return "";
    const min = Math.min(...losses);
    const max = Math.max(...losses);
    const span = max - min || 1;
    return losses
      .map((v, i) => {
        const x = pad + (i / Math.max(losses.length - 1, 1)) * (w - pad * 2);
        const y = pad + (1 - (v - min) / span) * (h - pad * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  })();

  return (
    <div className="loss-panel">
      <div className="loss-controls">
        <label className="loss-field">
          Steps
          <input
            type="number"
            min={1}
            step={1}
            value={steps}
            onChange={(e) => setSteps(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
          />
        </label>
        <label className="loss-field">
          LR
          <input
            type="number"
            min={0}
            step="any"
            value={lr}
            onChange={(e) => setLr(Number(e.target.value) || 0)}
          />
        </label>
        <button
          className="btn btn-sm"
          onClick={() => train(steps, lr)}
          disabled={status === "running"}
        >
          {status === "running" ? "Training…" : `Train ${steps} steps`}
        </button>
      </div>
      {losses.length === 0 ? (
        <div className="port-kind" style={{ padding: 8 }}>
          No loss history. Add a loss node (e.g. cross_entropy_loss) + a data
          loader, set steps / LR above, then click Train.
        </div>
      ) : (
        <>
          <svg viewBox={`0 0 ${w} ${h}`} className="loss-svg" preserveAspectRatio="none">
            <polyline
              points={points}
              fill="none"
              stroke="var(--color-primary)"
              strokeWidth="2"
            />
          </svg>
          <div className="loss-meta">
            <span>steps: {losses.length}</span>
            <span>last loss: {losses[losses.length - 1]?.toFixed(4)}</span>
          </div>
        </>
      )}
    </div>
  );
}
