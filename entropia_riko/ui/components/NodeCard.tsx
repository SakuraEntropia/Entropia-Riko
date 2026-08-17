/** Custom React Flow node card: title, ports, state, result (UI_STANDARD.md). */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { ThnNodeData } from "../store/graphStore";
import { useGraphStore } from "../store/graphStore";

export function NodeCard({ id, data, selected }: NodeProps) {
  // React Flow passes generic NodeProps; narrow it to our typed node data.
  const d = data as unknown as ThnNodeData;
  const result = useGraphStore((s) => s.results[id]);

  // Layout: header, port rows (inputs on the left with target handles, outputs
  // on the right with source handles), then an error line or per-output results.
  return (
    <div className={`thn-node ${selected ? "selected" : ""}`}>
      <div className="header">{d.label}</div>
      <div className="ports">
        {d.inputs.map((p) => (
          <div className="port in" key={p.name}>
            <Handle
              type="target"
              position={Position.Left}
              id={p.name}
              style={{ left: -5 }}
            />
            <span className="label">{p.label}</span>
          </div>
        ))}
        {d.outputs.map((p) => (
          <div className="port out" key={p.name}>
            <span className="label">{p.label}</span>
            <Handle
              type="source"
              position={Position.Right}
              id={p.name}
              style={{ right: -5 }}
            />
          </div>
        ))}
      </div>
      {d.hasError && (
        <div
          className="result"
          style={{ color: "var(--color-error)" }}
        >
          ⚠ {d.errorMsg || "error"}
        </div>
      )}
      {result &&
        Object.entries(result).map(([k, v]) => (
          <div className="result" key={k}>
            {k} = {v.summary}
          </div>
        ))}
    </div>
  );
}
