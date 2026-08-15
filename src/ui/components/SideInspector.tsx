/** Right side inspector: a left tag strip + tabbed content (Summary / Params /
 * Inputs / Outputs / Preview). */
import { useState, type CSSProperties } from "react";
import { useGraphStore } from "../store/graphStore";

// Fixed tab order shown in the left tag strip; each id maps to a section below.
const TABS = [
  { id: "summary", label: "Summary", icon: "▤" },
  { id: "params", label: "Params", icon: "⚙" },
  { id: "inputs", label: "Inputs", icon: "→" },
  { id: "outputs", label: "Outputs", icon: "←" },
  { id: "preview", label: "Preview", icon: "◉" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export function SideInspector({ style }: { style?: CSSProperties }) {
  const selectedId = useGraphStore((s) => s.selectedNodeId);
  const nodes = useGraphStore((s) => s.nodes);
  const results = useGraphStore((s) => s.results);
  const updateParam = useGraphStore((s) => s.updateParam);
  const removeNode = useGraphStore((s) => s.removeNode);
  const nodeDefs = useGraphStore((s) => s.nodeDefs);
  const [active, setActive] = useState<TabId>("summary");

  // Resolve the selected node once so every tab shares the same lookup.
  const node = nodes.find((n) => n.id === selectedId);

  // The body shows only the active tab, or a hint when nothing is selected.
  return (
    <div className="side-inspector" style={style}>
      <div className="inspector-tags">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`inspector-tag ${active === t.id ? "active" : ""}`}
            onClick={() => setActive(t.id)}
            title={t.label}
          >
            <span className="inspector-tag-icon">{t.icon}</span>
            <span className="inspector-tag-label">{t.label}</span>
          </button>
        ))}
      </div>

      <div className="inspector-body">
        {!node ? (
          <div className="empty">Select a node to edit its parameters</div>
        ) : (
          <>
            {active === "summary" && (
              <SummarySection node={node} onDelete={() => removeNode(node.id)} />
            )}
            {active === "params" && (
              <ParamsSection
                node={node}
                def={nodeDefs.find((n) => n.type === node.data.type)}
                onParam={(name, value) => updateParam(node.id, name, value)}
              />
            )}
            {active === "inputs" && <PortsSection title="Inputs" ports={node.data.inputs} dir="in" />}
            {active === "outputs" && <PortsSection title="Outputs" ports={node.data.outputs} dir="out" />}
            {active === "preview" && <PreviewSection result={results[node.id]} />}
          </>
        )}
      </div>
    </div>
  );
}

function SummarySection({
  node,
  onDelete,
}: {
  node: { data: { label: string; type: string; category: string } };
  onDelete: () => void;
}) {
  const d = node.data;
  return (
    <div className="section">
      <h3>Summary</h3>
      <div>{d.label}</div>
      <div className="port-kind">
        {d.type} · {d.category}
      </div>
      <button
        className="btn"
        style={{ marginTop: 8, color: "var(--color-error)", borderColor: "var(--color-error)" }}
        onClick={onDelete}
      >
        Delete
      </button>
    </div>
  );
}

function ParamsSection({
  node,
  def,
  onParam,
}: {
  node: { data: { params: Record<string, unknown> } };
  def: { parameters: { name: string; label: string; dtype?: string }[] } | undefined;
  onParam: (name: string, value: unknown) => void;
}) {
  return (
    <div className="section">
      <h3>Parameters</h3>
      {def && def.parameters.length > 0 ? (
        def.parameters.map((p) => (
          <div className="field" key={p.name}>
            <label>{p.label}</label>
            <input
              value={String(node.data.params[p.name] ?? "")}
              onChange={(e) => {
                // Coerce numeric fields back to Number so params keep their dtype.
                const v =
                  p.dtype === "float" || p.dtype === "int"
                    ? Number(e.target.value)
                    : e.target.value;
                onParam(p.name, v);
              }}
            />
          </div>
        ))
      ) : (
        <div className="port-kind">No parameters</div>
      )}
    </div>
  );
}

function PortsSection({
  title,
  ports,
  dir,
}: {
  title: string;
  ports: { name: string; label: string; dataKind: string }[];
  dir: string;
}) {
  return (
    <div className="section">
      <h3>{title}</h3>
      {ports.length === 0 ? (
        <div className="port-kind">None</div>
      ) : (
        ports.map((p) => (
          <div className="port-row" key={p.name}>
            <span>{p.label}</span>
            <span className="port-kind">
              {dir} · {p.dataKind}
            </span>
          </div>
        ))
      )}
    </div>
  );
}

function PreviewSection({ result }: { result: Record<string, {
  shape: number[]; dtype: string; device: string; summary: string;
  dataPreview: string; image?: string;
}> | undefined }) {
  // For each output key: shape/dtype/device, then an image or a text summary
  // (truncated with an ellipsis when the preview text is long).
  return (
    <div className="section">
      <h3>Preview</h3>
      {result ? (
        <div className="preview">
          {Object.entries(result).map(([k, tv]) => (
            <div key={k} style={{ marginBottom: 6 }}>
              <div>
                <strong>{k}</strong>:{" "}
                {tv.shape.length === 0 ? "scalar" : tv.shape.join("×")}{" "}
                {tv.dtype} · {tv.device}
              </div>
              {tv.image ? (
                <img
                  src={tv.image}
                  alt="preview"
                  style={{ maxWidth: "100%", marginTop: 4, borderRadius: 4, border: "1px solid var(--color-border)" }}
                />
              ) : (
                <>
                  <div>{tv.summary}</div>
                  {tv.dataPreview && (
                    <div style={{ color: "var(--color-text-secondary)", whiteSpace: "pre-wrap" }}>
                      {tv.dataPreview}
                      {tv.dataPreview.length >= 120 && "…"}
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="port-kind">Not executed</div>
      )}
    </div>
  );
}
