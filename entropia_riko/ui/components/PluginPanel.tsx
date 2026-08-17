/** Plugin manager panel: list installed plugins, enable/disable each with a
 * checkbox, and load new plugins from a `.py` file via the top-right "+". */
import { useEffect, useRef, useState } from "react";
import { useGraphStore } from "../store/graphStore";

interface PluginInfo {
  name: string;
  dir?: string;
  version?: string;
  description?: string;
  author?: string;
  status: string;
  error?: string;
  enabled?: boolean;
  nodes?: string[];
}

export function PluginPanel() {
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const pushToast = useGraphStore((s) => s.pushToast);

  // Load the installed-plugin list from the API server.
  const refresh = async () => {
    try {
      const r = await fetch("/api/plugins");
      const d = await r.json();
      setPlugins(Array.isArray(d.plugins) ? d.plugins : []);
    } catch {
      setPlugins([]);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  // Enable/disable a plugin, then reload node defs so its nodes enter/leave the library.
  const toggle = async (p: PluginInfo, enabled: boolean) => {
    setBusy(true);
    try {
      const r = await fetch("/api/plugins/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: p.dir ?? p.name, enabled }),
      });
      const d = await r.json();
      if (d.status === "success") {
        setPlugins(Array.isArray(d.plugins) ? d.plugins : []);
        useGraphStore.getState().loadNodeDefs(); // refresh the node library
        pushToast("success", `${p.name} ${enabled ? "enabled" : "disabled"}`);
      } else {
        pushToast("error", d.error ?? "toggle failed");
      }
    } catch (e) {
      pushToast("error", e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // Read the selected .py file's text (via FileReader) and upload it as a new plugin.
  const onPickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    // Reset the input so the same file can be loaded again later.
    e.target.value = "";
    if (!f) return;
    const reader = new FileReader();
    reader.onload = async () => {
      // FileReader returns the file contents as a string; fall back to "".
      const code = String(reader.result ?? "");
      const name = f.name.replace(/\.py$/i, "") || "plugin";
      setBusy(true);
      try {
        const r = await fetch("/api/plugins/upload", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, code }),
        });
        const d = await r.json();
        if (d.status === "success") {
          setPlugins(Array.isArray(d.plugins) ? d.plugins : []);
          useGraphStore.getState().loadNodeDefs();
          pushToast("success", `Loaded plugin ${name}`);
        } else {
          pushToast("error", d.error ?? "upload failed");
        }
      } catch (err) {
        pushToast("error", err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    };
    reader.readAsText(f);
  };

  return (
    <div className="plugin-panel">
      <div className="panel-head">
        <span className="panel-head-title">Plugins</span>
        <span className="spacer" />
        <button
          className="panel-btn"
          title="Load plugin from a .py file"
          disabled={busy}
          onClick={() => fileRef.current?.click()}
        >
          +
        </button>
        <button className="panel-btn" title="Refresh" onClick={refresh}>
          ↻
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".py,text/x-python"
          style={{ display: "none" }}
          onChange={onPickFile}
        />
      </div>
      <div className="plugin-list">
        {plugins.length === 0 && <div className="prefs-empty">No plugins installed.</div>}
        {plugins.map((p) => (
          <div key={p.name} className="plugin-item">
            <div className="plugin-item-main">
              <span className="plugin-item-name">
                {p.name} {p.version ? `v${p.version}` : ""}
              </span>
              <span className={`prefs-plugin-status ${p.status}`}>{p.status}</span>
              {p.description && <span className="plugin-item-desc">{p.description}</span>}
              {p.error && <span className="prefs-plugin-error">{p.error}</span>}
            </div>
            <label className="plugin-toggle" title={p.enabled ? "Disable" : "Enable"}>
              <input
                type="checkbox"
                checked={!!p.enabled}
                disabled={busy || p.status === "error"}
                onChange={(e) => toggle(p, e.target.checked)}
              />
              <span className="plugin-toggle-track" />
            </label>
          </div>
        ))}
      </div>
    </div>
  );
}
