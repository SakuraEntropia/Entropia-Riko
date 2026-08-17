/** Reusable Windows-style file picker (promise API). Used by every import/export
 * site: call `openFilePicker("import" | "export")` and await the selected path. */
import { useEffect, useState } from "react";
import { useGraphStore } from "../store/graphStore";
import { FloatingWindow } from "./FloatingWindow";

interface FsEntry {
  name: string;
  path: string;
  type: "dir" | "file";
  size: number;
}

// Singleton resolver for the currently-open picker's promise, plus a default
// filename prefill for "save" mode that must be set before the host reads it.
let resolver: ((v: string | null) => void) | null = null;
let pickerDefaultName = "";

export function openFilePicker(
  mode: "import" | "export" | "save",
  options?: { defaultName?: string }
): Promise<string | null> {
  pickerDefaultName = options?.defaultName ?? "";
  // Ask the global store to open the picker host window in the given mode.
  useGraphStore.getState().setFilePicker(true, mode);
  // Stash the resolver so FilePickerHost can settle this promise later.
  return new Promise((res) => {
    resolver = res;
  });
}

function resolve(value: string | null) {
  // Close the host, settle the pending promise, and reset the singleton.
  useGraphStore.getState().setFilePicker(false, "import");
  resolver?.(value);
  resolver = null;
}

const RECENT_KEY = "entropia_riko_recent_folders";

function loadRecent(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    const arr = raw ? (JSON.parse(raw) as unknown) : [];
    // Sanity-check the parsed payload before returning it as a string list.
    return Array.isArray(arr) ? (arr as string[]).filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

function saveRecent(p: string) {
  try {
    // Newest-first with duplicates removed, capped at 8 entries.
    const next = [p, ...loadRecent().filter((x) => x !== p)].slice(0, 8);
    localStorage.setItem(RECENT_KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
}

export function FilePickerHost() {
  const open = useGraphStore((s) => s.filePickerOpen);
  const mode = useGraphStore((s) => s.filePickerMode);
  const pushToast = useGraphStore((s) => s.pushToast);
  const [path, setPath] = useState("~");
  const [entries, setEntries] = useState<FsEntry[]>([]);
  const [sel, setSel] = useState<FsEntry | null>(null);
  const [workRoot, setWorkRoot] = useState("~");
  const [recent, setRecent] = useState<string[]>(loadRecent());
  const [back, setBack] = useState<string[]>([]);
  const [fwd, setFwd] = useState<string[]>([]);
  const [saveName, setSaveName] = useState("");

  const navigate = async (p: string, pushHistory = true) => {
    const r = await fetch(`/api/fs/list?path=${encodeURIComponent(p)}`);
    const d = await r.json();
    if (d.status === "success") {
      if (pushHistory) {
        // Record the previous dir for Back and clear any Forward history.
        setBack((b) => [...b, path]);
        setFwd([]);
      }
      setPath(d.path);
      setEntries(d.entries ?? []);
      setSel(null);
      // Refresh the recent list from localStorage after each successful visit.
      saveRecent(d.path);
      setRecent(loadRecent());
    } else {
      pushToast("error", d.error ?? "list failed");
    }
  };

  useEffect(() => {
    if (open) {
      if (mode === "save") setSaveName(pickerDefaultName);
      // Reset to home (without pushing history) each time the picker opens.
      navigate("~", false);
      // Fetch the project working folder for the "Working Folder" shortcut.
      fetch("/api/project/tree")
        .then((r) => r.json())
        .then((d) => {
          if (typeof d.root === "string") setWorkRoot(d.root);
        })
        .catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  const quick = [
    { label: "Home", path: "~" },
    { label: "Desktop", path: "~/Desktop" },
    { label: "Documents", path: "~/Documents" },
    { label: "Downloads", path: "~/Downloads" },
    { label: "Working Folder", path: workRoot },
  ];

  const goBack = () => {
    const prev = back[back.length - 1];
    if (!prev) return;
    // Pop the back stack and push the current dir onto the forward stack.
    setBack(back.slice(0, -1));
    setFwd((f) => [path, ...f]);
    navigate(prev, false);
  };
  const goFwd = () => {
    const next = fwd[0];
    if (!next) return;
    // Shift the forward stack and push the current dir onto the back stack.
    setFwd(fwd.slice(1));
    setBack((b) => [...b, path]);
    navigate(next, false);
  };

  const result = () => {
    // Save mode: join the current directory with the typed (or default) filename.
    if (mode === "save") {
      const name = saveName.trim() || "untitled";
      return path.endsWith("/") ? path + name : path + "/" + name;
    }
    // Export mode: prefer a selected folder, else the current directory.
    if (mode === "export") {
      return sel && sel.type === "dir" ? sel.path : path;
    }
    // Import mode: resolve the selected entry, falling back to the directory.
    return sel ? sel.path : path;
  };

  return (
    <FloatingWindow
      title={mode === "import" ? "Import File / Folder" : mode === "export" ? "Export To Folder" : "Save File"}
      onClose={() => resolve(null)}
      width={780}
      zIndex={1002}
    >
      <div className="picker-inner">
        <div className="picker-side">
          <div className="picker-side-title">Quick access</div>
          {quick.map((q) => (
            <div key={q.label} className="picker-side-item" onClick={() => navigate(q.path)}>
              <span className="picker-side-icon">▦</span>
              {q.label}
            </div>
          ))}
          <div className="picker-side-title" style={{ marginTop: 12 }}>Recent</div>
          {recent.map((r) => (
            <div key={r} className="picker-side-item" title={r} onClick={() => navigate(r)}>
              <span className="picker-side-icon">🕘</span>
              <span className="picker-side-name">{r}</span>
            </div>
          ))}
        </div>
        <div className="picker-main">
          <div className="picker-topbar">
            <button className="panel-btn" onClick={goBack} title="Back">←</button>
            <button className="panel-btn" onClick={goFwd} title="Forward">→</button>
            <button className="panel-btn" onClick={() => navigate(path, false)} title="Refresh">↻</button>
            <input
              className="prefs-input"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") navigate(path);
              }}
            />
          </div>
          <div className="picker-list">
            {entries.map((e) => (
              <div
                key={e.path}
                className={`picker-row ${sel?.path === e.path ? "sel" : ""}`}
                onClick={() => setSel(e)}
                onDoubleClick={() => e.type === "dir" && navigate(e.path)}
              >
                <span className="explorer-icon">{e.type === "dir" ? "▦" : "▤"}</span>
                <span className="explorer-name">{e.name}</span>
                <span className="explorer-size">{e.type === "file" ? `${Math.round(e.size / 1024)} KB` : ""}</span>
              </div>
            ))}
            {entries.length === 0 && <div className="welcome-empty">Empty folder.</div>}
          </div>
          <div className="picker-bottombar">
            <span className="picker-filename-label">File name:</span>
            <input
              className="prefs-input"
              value={mode === "save" ? saveName : sel?.name ?? ""}
              onChange={(e) => {
                if (mode === "save") setSaveName(e.target.value);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") resolve(result());
                if (e.key === "Escape") resolve(null);
              }}
              readOnly={mode !== "save"}
              placeholder={mode === "save" ? "name.riko / name.ric" : "(select a file or folder)"}
            />
            <button className="btn btn-primary" onClick={() => resolve(result())}>
              Save
            </button>
            <button className="btn" onClick={() => resolve(null)}>Cancel</button>
          </div>
        </div>
      </div>
    </FloatingWindow>
  );
}
