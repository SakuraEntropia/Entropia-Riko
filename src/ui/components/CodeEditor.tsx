/** Code editor workspace window — Notepad-style with File/Edit menus plus a
 * quick-action toolbar (New / Open / Save / Undo / Redo / Cut / Copy / Paste). */
import { useRef, useState } from "react";
import { useGraphStore } from "../store/graphStore";

export function CodeEditor() {
  const codeBuffer = useGraphStore((s) => s.codeBuffer);
  const setCodeBuffer = useGraphStore((s) => s.setCodeBuffer);
  const generatePythonCode = useGraphStore((s) => s.generatePythonCode);
  const pushToast = useGraphStore((s) => s.pushToast);
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [openMenu, setOpenMenu] = useState<string | null>(null);

  const focus = () => ref.current?.focus();
  // Drive Undo/Redo/Cut/Paste via execCommand on the focused textarea.
  const exec = (cmd: string) => {
    focus();
    document.execCommand(cmd);
  };
  const copy = async () => {
    try {
      // Prefer the async Clipboard API; fall back to execCommand if unavailable.
      await navigator.clipboard.writeText(codeBuffer);
      pushToast("success", "Copied");
    } catch {
      exec("copy");
    }
  };
  const selectAll = () => {
    focus();
    ref.current?.select();
  };
  const save = () => {
    // Trigger a browser download of the buffer as code.py via a temp object URL.
    const blob = new Blob([codeBuffer], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "code.py";
    a.click();
    URL.revokeObjectURL(url);
  };
  const clear = () => setCodeBuffer("");
  const onOpen = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    const reader = new FileReader();
    // Read the chosen file as text and drop it into the code buffer.
    reader.onload = () => setCodeBuffer(String(reader.result ?? ""));
    reader.readAsText(f);
  };

  // Declarative menu definitions: File / Edit dropdowns for the menubar.
  const menus = [
    {
      label: "File",
      items: [
        { label: "New", run: clear },
        { label: "Open…", run: () => fileRef.current?.click() },
        { label: "Generate from Graph", run: generatePythonCode },
        { label: "Save .py", run: save },
      ],
    },
    {
      label: "Edit",
      items: [
        { label: "Undo", run: () => exec("undo") },
        { label: "Redo", run: () => exec("redo") },
        { label: "Cut", run: () => exec("cut") },
        { label: "Copy", run: copy },
        { label: "Paste", run: () => exec("paste") },
        { label: "Select All", run: selectAll },
        { label: "Clear", run: clear },
      ],
    },
  ];

  // Declarative toolbar buttons; icon/title/action rendered in a single row.
  const toolbar: { icon: string; title: string; run: () => void }[] = [
    { icon: "＋", title: "New", run: clear },
    { icon: "📂", title: "Open", run: () => fileRef.current?.click() },
    { icon: "💾", title: "Save .py", run: save },
    { icon: "↶", title: "Undo", run: () => exec("undo") },
    { icon: "↷", title: "Redo", run: () => exec("redo") },
    { icon: "✂", title: "Cut", run: () => exec("cut") },
    { icon: "⧉", title: "Copy", run: copy },
    { icon: "📋", title: "Paste", run: () => exec("paste") },
    { icon: "▦", title: "Select All", run: selectAll },
  ];

  return (
    <div className="code-editor">
      <div className="code-menubar">
        {menus.map((m) => (
          <div
            key={m.label}
            className={`code-menu-item ${openMenu === m.label ? "active" : ""}`}
            onClick={() => setOpenMenu(openMenu === m.label ? null : m.label)}
          >
            {m.label}
            {openMenu === m.label && (
              <div className="code-menu-dropdown" onClick={(e) => e.stopPropagation()}>
                {m.items.map((it, i) => (
                  <div
                    key={i}
                    className="code-menu-entry"
                    onClick={() => {
                      it.run();
                      setOpenMenu(null);
                    }}
                  >
                    {it.label}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="code-toolbar">
        {toolbar.map((b, i) => (
          <button key={i} className="panel-btn" title={b.title} onClick={b.run}>
            {b.icon}
          </button>
        ))}
      </div>
      <textarea
        ref={ref}
        className="code-editor-area"
        value={codeBuffer}
        onChange={(e) => setCodeBuffer(e.target.value)}
        placeholder="// Right-click a .riko file → Preview PyTorch Code, or File → Generate from Graph."
        spellCheck={false}
      />
      <input
        ref={fileRef}
        type="file"
        accept=".py,.txt,.riko,.ric,text/plain,text/x-python"
        style={{ display: "none" }}
        onChange={onOpen}
      />
    </div>
  );
}
