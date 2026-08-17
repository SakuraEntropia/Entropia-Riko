/** "Import Working Folder" dialog — sets the working directory the mini file
 * manager maps (and creates a `.riko` cache folder inside it). */
import { useState } from "react";
import { useGraphStore } from "../store/graphStore";
import { FloatingWindow } from "./FloatingWindow";

export function ImportFolderPanel({ onClose }: { onClose: () => void }) {
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const pushToast = useGraphStore((s) => s.pushToast);

  const submit = async () => {
    const p = path.trim();
    if (!p) return;
    setBusy(true);
    try {
      const r = await fetch("/api/project/set_root", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: p }),
      });
      const d = await r.json();
      if (d.status === "success") {
        useGraphStore.getState().bumpProject();
        pushToast("success", `Working folder: ${d.root}`);
        onClose();
      } else {
        pushToast("error", d.error ?? "import failed");
      }
    } catch (e) {
      pushToast("error", e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <FloatingWindow title="Import Working Folder" onClose={onClose} width={480}>
      <div className="import-folder-body">
        <p className="prefs-desc">
          Point the file manager at a local folder. A <code>.riko</code> cache
          folder is created inside it for the tool's own state.
        </p>
        <input
          autoFocus
          className="prefs-input"
          placeholder="Absolute path, e.g. /Users/me/project"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
            if (e.key === "Escape") onClose();
          }}
        />
        <div className="prefs-row" style={{ marginTop: 12 }}>
          <button className="btn btn-primary" disabled={busy || !path.trim()} onClick={submit}>
            Import
          </button>
          <button className="btn" onClick={onClose}>Cancel</button>
        </div>
      </div>
    </FloatingWindow>
  );
}
