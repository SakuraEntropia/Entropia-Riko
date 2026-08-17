/** Toolbar: Run / Save / Load + status badge (UI_STANDARD.md). */
import { useRef } from "react";
import { useGraphStore } from "../store/graphStore";

export function Toolbar() {
  const status = useGraphStore((s) => s.status);
  const run = useGraphStore((s) => s.run);
  const save = useGraphStore((s) => s.save);
  const load = useGraphStore((s) => s.load);
  const exportPython = useGraphStore((s) => s.exportPython);
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <div className="toolbar">
      <h1>Entropia Riko</h1>
      <button
        className="btn btn-primary"
        onClick={run}
        disabled={status === "running"}
      >
        {status === "running" ? "Running…" : "Inference"}
      </button>
      <button
        className="btn"
        onClick={() => {
          useGraphStore.getState().log("训练模式：单步执行...");
          run();
        }}
        disabled={status === "running"}
        style={{ borderColor: "var(--color-success)", color: "var(--color-success)" }}
      >
        Train
      </button>
      <button className="btn" onClick={exportPython} disabled={status === "running"}>
        Export .py
      </button>
      <button
        className="btn"
        onClick={() => useGraphStore.getState().addNode("mnist_loader", { x: 300, y: 200 })}
        disabled={status === "running"}
      >
        Load Data
      </button>
      <button className="btn" onClick={save} disabled={status === "running"}>
        Save
      </button>
      <button
        className="btn"
        onClick={() => fileRef.current?.click()}
        disabled={status === "running"}
      >
        Load
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".json,application/json"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) load(f);
          e.target.value = "";
        }}
      />
      <div className="spacer" />
      <span className={`status-badge ${status}`}>{status}</span>
    </div>
  );
}
