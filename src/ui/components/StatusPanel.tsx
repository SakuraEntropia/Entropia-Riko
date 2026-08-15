/** Bottom status panel: logs + queue (UI_STANDARD.md). */
import { useState, type CSSProperties } from "react";
import { useGraphStore } from "../store/graphStore";

export function StatusPanel({ style }: { style?: CSSProperties }) {
  const [tab, setTab] = useState<"logs" | "queue">("logs");
  const logs = useGraphStore((s) => s.logs);
  const nodes = useGraphStore((s) => s.nodes);
  const status = useGraphStore((s) => s.status);

  return (
    <div className="status-panel" style={style}>
      <div className="tabs">
        <div
          className={`tab ${tab === "logs" ? "active" : ""}`}
          onClick={() => setTab("logs")}
        >
          Logs
        </div>
        <div
          className={`tab ${tab === "queue" ? "active" : ""}`}
          onClick={() => setTab("queue")}
        >
          Queue
        </div>
      </div>
      <div className="content">
        {tab === "logs" ? (
          logs.length === 0 ? (
            <div className="log-line">No logs yet. Add nodes and click Run.</div>
          ) : (
            logs.map((l, i) => (
              <div className="log-line" key={i}>
                {l}
              </div>
            ))
          )
        ) : nodes.length === 0 ? (
          <div className="log-line">Graph is empty</div>
        ) : (
          nodes.map((n) => (
            <div className="queue-item" key={n.id}>
              <span>{n.data.label}</span>
              <span className="port-kind">{status}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
