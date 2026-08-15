/** Left node library: nodes grouped by category with collapsible sections. */
import { useState, useMemo } from "react";
import { useGraphStore } from "../store/graphStore";

export function NodeLibrary() {
  const [query, setQuery] = useState("");
  const nodeDefs = useGraphStore((s) => s.nodeDefs);
  const addNode = useGraphStore((s) => s.addNode);
  const addFrame = useGraphStore((s) => s.addFrame);

  // Filter matching defs and bucket them by category for the collapsible list.
  const grouped = useMemo(() => {
    const filtered = nodeDefs.filter(
      (n) =>
        n.label.toLowerCase().includes(query.toLowerCase()) ||
        n.type.toLowerCase().includes(query.toLowerCase())
    );
    const map: Record<string, typeof nodeDefs> = {};
    for (const n of filtered) {
      // Normalize a missing category so every node lands under a section.
      const cat = n.category || "Other";
      (map[cat] ??= []).push(n);
    }
    return map;
  }, [nodeDefs, query]);

  // Sort category headings alphabetically for a stable, predictable order.
  const categories = Object.keys(grouped).sort();

  // New nodes are dropped near the frame origin with a small jitter so they
  // don't stack exactly on top of each other.
  return (
    <div className="node-library">
      <div className="search">
        <input
          placeholder="Search nodes…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          className="btn btn-sm"
          style={{ width: "100%", marginTop: 4 }}
          onClick={() => addFrame("Frame", { x: 200, y: 120 })}
        >
          ＋ Frame
        </button>
      </div>
      <div className="node-list">
        {categories.map((cat) => (
          <CategorySection
            key={cat}
            category={cat}
            nodes={grouped[cat]}
            onAdd={(type) => addNode(type, { x: 200 + Math.random() * 200, y: 120 + Math.random() * 100 })}
          />
        ))}
        {categories.length === 0 && (
          <div className="category" style={{ padding: 8 }}>No matching nodes</div>
        )}
        {nodeDefs.length < 10 && (
          <div style={{ padding: "8px", color: "var(--color-error)", fontSize: 12, lineHeight: 1.5 }}>
            ⚠ 仅 {nodeDefs.length} 个节点。请确保 API server 已启动。
          </div>
        )}
      </div>
    </div>
  );
}

function CategorySection({
  category,
  nodes,
  onAdd,
}: {
  category: string;
  nodes: { type: string; label: string; category: string }[];
  onAdd: (type: string) => void;
}) {
  // Each section keeps its own open/closed state; all start expanded.
  const [open, setOpen] = useState(true);
  return (
    <div className="node-category">
      <div
        className="category-header"
        onClick={() => setOpen(!open)}
      >
        <span className="category-toggle">{open ? "▼" : "▶"}</span>
        <span className="category-name">{category}</span>
        <span className="category-count">{nodes.length}</span>
      </div>
      {open && nodes.map((n) => (
        <div
          key={n.type}
          className="node-item"
          onClick={() => onAdd(n.type)}
        >
          <div className="name">{n.label}</div>
        </div>
      ))}
    </div>
  );
}
