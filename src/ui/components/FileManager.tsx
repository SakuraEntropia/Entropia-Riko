/** Asset Library — a clean tree of .riko/.ric assets (workflows + examples):
 * search, open, expand-full-nodes, preview PyTorch code, and save current. */
import { useEffect, useMemo, useState } from "react";
import { useGraphStore, type FileInfo } from "../store/graphStore";

export function FileManager() {
  const fileList = useGraphStore((s) => s.fileList);
  const activeFileName = useGraphStore((s) => s.activeFileName);
  const refreshFiles = useGraphStore((s) => s.refreshFiles);
  const openFile = useGraphStore((s) => s.openFile);
  const saveFileToDisk = useGraphStore((s) => s.saveFileToDisk);
  const importFileInline = useGraphStore((s) => s.importFileInline);
  const previewAssetCode = useGraphStore((s) => s.previewAssetCode);

  const [query, setQuery] = useState("");

  useEffect(() => {
    refreshFiles();
  }, [refreshFiles]);

  const grouped = useMemo(() => {
    const q = query.trim().toLowerCase();
    const map: Record<string, FileInfo[]> = {};
    const order: string[] = [];
    for (const f of fileList) {
      // Keep entries whose name or path matches the search query (when set).
      if (q && !f.name.toLowerCase().includes(q) && !f.path.toLowerCase().includes(q)) continue;
      // Group by the top-level directory segment; bare files land in "(root)".
      const top = f.path.includes("/") ? f.path.slice(0, f.path.indexOf("/")) : "(root)";
      if (!(top in map)) {
        map[top] = [];
        order.push(top);
      }
      map[top].push(f);
    }
    // Sort group names alphabetically for stable rendering.
    order.sort();
    return { map, order };
  }, [fileList, query]);

  const saveCurrent = () => {
    // Prompt for a name, then persist the current graph under that filename.
    const name = window.prompt("Save current graph as", activeFileName ?? "untitled");
    if (name && name.trim()) saveFileToDisk(name.trim());
  };

  return (
    <div className="file-manager">
      <div className="panel-head">
        <span className="panel-head-title">Asset Library</span>
        <span className="spacer" />
        <button className="panel-btn" title="Save current graph" onClick={saveCurrent}>💾</button>
        <button className="panel-btn" title="Refresh" onClick={refreshFiles}>↻</button>
      </div>
      <div className="asset-search">
        <input
          placeholder="Search assets…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      <div className="fm-tree">
        {grouped.order.map((dir) => (
          <div className="fm-group" key={dir}>
            <div className="fm-group-header">{dir}</div>
            {grouped.map[dir].map((f) => (
              <div
                key={f.path}
                className={`fm-file ${activeFileName === f.name ? "active" : ""}`}
                onClick={() => openFile(f.path)}
                title={f.path}
              >
                <span className="fm-toggle">▤</span>
                <span className="fm-name">{f.name}</span>
                {f.format === "binary" && (
                  <span className="fm-format-badge" title="binary .ric">BIN</span>
                )}
                <button
                  className="fm-btn"
                  title="Preview PyTorch code"
                  onClick={(e) => {
                    e.stopPropagation();
                    previewAssetCode(f.path);
                  }}
                >
                  ⤡
                </button>
                <button
                  className="fm-btn"
                  title="Expand full nodes into canvas"
                  onClick={(e) => {
                    e.stopPropagation();
                    importFileInline(f.path);
                  }}
                >
                  ⤢
                </button>
              </div>
            ))}
          </div>
        ))}
        {fileList.length === 0 && (
          <div className="category" style={{ padding: 8 }}>No .riko files found.</div>
        )}
        {fileList.length > 0 && grouped.order.length === 0 && (
          <div className="category" style={{ padding: 8 }}>No matches for “{query}”.</div>
        )}
      </div>
    </div>
  );
}
