/** Welcome screen: header image (hero) + logo + version badge over two columns
 * (New File presets | Recent Files). */
import { useEffect, useRef, useState } from "react";
import { useGraphStore, type FileInfo } from "../store/graphStore";
import { APP_VERSION } from "../version";

export function WelcomePanel({ onClose }: { onClose: () => void }) {
  const recentFiles = useGraphStore((s) => s.recentFiles);
  const openFile = useGraphStore((s) => s.openFile);
  const newBlank = useGraphStore((s) => s.newBlank);
  const load = useGraphStore((s) => s.load);
  const pushToast = useGraphStore((s) => s.pushToast);
  const [presets, setPresets] = useState<FileInfo[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  // Fetch the preset list once; keep only the bundled "examples" and sort them.
  useEffect(() => {
    fetch("/api/files")
      .then((r) => r.json())
      .then((d) => {
        if (Array.isArray(d.files)) {
          setPresets(
            (d.files as FileInfo[])
              .filter((f) => f.path.startsWith("examples"))
              .sort((a, b) => a.name.localeCompare(b.name))
          );
        }
      })
      .catch(() => {
        /* no server: presets empty */
      });
  }, []);

  // Strip directories and the .riko extension for the recent-files display label.
  const basename = (p: string) => p.split("/").pop()?.replace(/\.riko$/, "") ?? p;

  const choose = (fn: () => void) => {
    fn();
    onClose();
  };

  // Reopen the most recent file, or warn when there is no session to recover.
  const recover = () => {
    const last = recentFiles[0];
    if (last) choose(() => openFile(last));
    else pushToast("error", "No previous session to recover");
  };

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    // Reset the input so re-picking the same file still fires onChange.
    e.target.value = "";
    if (f) {
      load(f);
      onClose();
    }
  };

  return (
    <div className="welcome-overlay">
      <div className="welcome">
        {/* Header-image slot: replace public/brand/hero.jpg (JPEG bitmap, 960×220) in place. */}
        <div className="welcome-hero">
          <img
            className="welcome-logo"
            src="/brand/logo.svg"
            alt="Entropia Riko logo"
            title="Logo slot — replace public/brand/logo.svg"
          />
          <span className="welcome-version">v{APP_VERSION}</span>
        </div>
        <div className="welcome-header">
          <span className="welcome-title">Entropia Riko</span>
          <button className="panel-btn" title="Close" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="welcome-body">
          <div className="welcome-col">
            <div className="welcome-col-title">New File</div>
            <button className="welcome-item welcome-blank" onClick={() => choose(newBlank)}>
              <span className="welcome-item-icon">⬚</span>
              <span>General</span>
              <span className="welcome-item-hint">blank graph</span>
            </button>
            <div className="welcome-section-label">Presets</div>
            <div className="welcome-scroll">
              {presets.length === 0 && <div className="welcome-empty">No presets (start the API server)</div>}
              {presets.map((p) => (
                <div key={p.path} className="welcome-item" onClick={() => choose(() => openFile(p.path))}>
                  <span className="welcome-item-icon">▦</span>
                  <span>{p.name}</span>
                  <span className="welcome-item-hint">{p.imports.length ? `imports ${p.imports.length}` : ""}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="welcome-col">
            <div className="welcome-col-title">Recent Files</div>
            <div className="welcome-scroll">
              {recentFiles.length === 0 ? (
                <div className="welcome-empty">No recent files yet</div>
              ) : (
                recentFiles.map((p) => (
                  <div key={p} className="welcome-item" onClick={() => choose(() => openFile(p))}>
                    <span className="welcome-item-icon">▤</span>
                    <span>{basename(p)}</span>
                    <span className="welcome-item-hint">{p}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
        <div className="welcome-footer">
          <button className="welcome-footer-btn" onClick={recover}>
            ↩ Recover Last Session
          </button>
          <button className="welcome-footer-btn" onClick={() => fileRef.current?.click()}>
            📂 Open
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".riko,.ric,.json,application/json,application/octet-stream"
            style={{ display: "none" }}
            onChange={onPick}
          />
        </div>
      </div>
    </div>
  );
}
