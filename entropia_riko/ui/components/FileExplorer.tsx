/** Built-in Windows-explorer-style file browser for importing/exporting files
 * and folders via copy (no direct browser download). */
import { useEffect, useState } from "react";
import { useGraphStore } from "../store/graphStore";

interface FsEntry {
  name: string;
  path: string;
  type: "dir" | "file";
  size: number;
}

interface TreeNode {
  name: string;
  path: string;
  type: "dir" | "file";
  children?: TreeNode[];
}

function flattenFiles(nodes: TreeNode[], out: string[] = []): string[] {
  for (const n of nodes) {
    if (n.type === "file") out.push(n.path);
    else if (n.children) flattenFiles(n.children, out);
  }
  return out;
}

export function FileExplorer({ onClose }: { onClose: () => void }) {
  const pushToast = useGraphStore((s) => s.pushToast);
  const bumpProject = useGraphStore((s) => s.bumpProject);
  const [path, setPath] = useState("~");
  const [entries, setEntries] = useState<FsEntry[]>([]);
  const [parent, setParent] = useState("");
  const [workFiles, setWorkFiles] = useState<string[]>([]);
  const [selWork, setSelWork] = useState("");

  const list = async (p: string) => {
    const r = await fetch(`/api/fs/list?path=${encodeURIComponent(p)}`);
    const d = await r.json();
    if (d.status === "success") {
      setPath(d.path);
      setParent(d.parent);
      setEntries(d.entries ?? []);
    } else {
      pushToast("error", d.error ?? "list failed");
    }
  };

  const loadWorkFiles = async () => {
    try {
      const r = await fetch("/api/project/tree");
      const d = await r.json();
      const files = flattenFiles(Array.isArray(d.tree) ? d.tree : []);
      setWorkFiles(files);
      if (files.length && !selWork) setSelWork(files[0]);
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    list("~");
    loadWorkFiles();
  }, []);

  const importEntry = async (e: FsEntry) => {
    const r = await fetch("/api/fs/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ src: e.path }),
    });
    const d = await r.json();
    if (d.status === "success") {
      pushToast("success", `Imported ${e.name}`);
      bumpProject();
      loadWorkFiles();
    } else {
      pushToast("error", d.error ?? "import failed");
    }
  };

  const exportHere = async () => {
    if (!selWork) return;
    const r = await fetch("/api/fs/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ src: selWork, dest: path }),
    });
    const d = await r.json();
    if (d.status === "success") pushToast("success", `Exported to ${d.path}`);
    else pushToast("error", d.error ?? "export failed");
  };

  return (
    <div className="prefs-overlay" onClick={onClose}>
      <div className="explorer" onClick={(e) => e.stopPropagation()}>
        <div className="prefs-header">
          <span className="prefs-title">File Explorer</span>
          <button className="panel-btn" onClick={onClose}>✕</button>
        </div>
        <div className="explorer-toolbar">
          <button className="panel-btn" title="Up" onClick={() => list(parent)}>↑</button>
          <input
            className="prefs-input"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") list(path);
            }}
          />
          <button className="panel-btn" onClick={() => list(path)}>Go</button>
        </div>
        <div className="explorer-body">
          <div className="explorer-list">
            {entries.map((e) => (
              <div
                key={e.path}
                className="explorer-row"
                onDoubleClick={() => e.type === "dir" && list(e.path)}
              >
                <span className="explorer-icon">{e.type === "dir" ? "▦" : "▤"}</span>
                <span className="explorer-name">{e.name}</span>
                <span className="explorer-size">
                  {e.type === "file" ? `${Math.round(e.size / 1024)} KB` : ""}
                </span>
                <button className="panel-btn" onClick={() => importEntry(e)} title="Import into working folder">
                  ⇥ Import
                </button>
              </div>
            ))}
            {entries.length === 0 && <div className="welcome-empty">Empty folder.</div>}
          </div>
          <div className="explorer-export">
            <div className="prefs-desc" style={{ margin: 0 }}>
              Export a working-folder file to the current folder:
            </div>
            <select className="prefs-input" value={selWork} onChange={(e) => setSelWork(e.target.value)}>
              {workFiles.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
            <button className="btn" disabled={!selWork} onClick={exportHere}>
              Export here
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
