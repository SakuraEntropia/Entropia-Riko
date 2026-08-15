/** Professional dropdown menu bar (replaces flat Toolbar). */
import { useState, useRef } from "react";
import { useGraphStore } from "../store/graphStore";
import { useThemeStore } from "../theme";
import { WorkspaceTabs } from "./WorkspaceTabs";
import { openFilePicker } from "./FilePicker";
import type { WorkspaceInstance } from "../areas";

interface MenuItem {
  label: string;
  onClick?: () => void;
  separator?: boolean;
  disabled?: boolean;
  submenu?: MenuItem[];
}

interface Menu {
  label: string;
  items: MenuItem[];
}

export function MenuBar({
  workspaces,
  activeWorkspaceId,
  onSwitchWorkspace,
  onAddWorkspace,
  onRemoveWorkspace,
  onRenameWorkspace,
  onDuplicateWorkspace,
  onMoveWorkspace,
}: {
  workspaces: WorkspaceInstance[];
  activeWorkspaceId: string;
  onSwitchWorkspace: (id: string) => void;
  onAddWorkspace: (presetId: string) => void;
  onRemoveWorkspace: (id: string) => void;
  onRenameWorkspace: (id: string, name: string) => void;
  onDuplicateWorkspace: (id: string) => void;
  onMoveWorkspace: (id: string, delta: number) => void;
}) {
  // Tracks which top-level menu (or the app logo) is currently open.
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const status = useGraphStore((s) => s.status);
  const run = useGraphStore((s) => s.run);
  const train = useGraphStore((s) => s.train);
  const save = useGraphStore((s) => s.save);
  const exportRiko = useGraphStore((s) => s.exportRiko);
  const exportBinary = useGraphStore((s) => s.exportBinary);
  const load = useGraphStore((s) => s.load);
  const exportPython = useGraphStore((s) => s.exportPython);
  const exportKeras = useGraphStore((s) => s.exportKeras);
  const newWorkflow = useGraphStore((s) => s.newWorkflow);
  const addNode = useGraphStore((s) => s.addNode);
  const pushToast = useGraphStore((s) => s.pushToast);
  const theme = useThemeStore((s) => s.theme);
  const setTheme = useThemeStore((s) => s.setTheme);
  // Hidden file input used by the legacy File menu for loading `.riko`/`.ric` files.
  const fileRef = useRef<HTMLInputElement>(null);
  // Disable run/export actions while a run or train is in progress.
  const busy = status === "running";

  // Open the file picker to choose a source, then copy it into the project.
  const importFile = async () => {
    const src = await openFilePicker("import");
    if (!src) return;
    const r = await fetch("/api/fs/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ src }),
    });
    const d = await r.json();
    if (d.status === "success") {
      pushToast("success", `Imported ${d.path}`);
      useGraphStore.getState().bumpProject();
    } else {
      pushToast("error", d.error ?? "import failed");
    }
  };

  // Export the currently open file to a user-chosen destination path.
  const exportFile = async () => {
    const src = useGraphStore.getState().activeFilePath;
    if (!src) {
      pushToast("error", "No open file to export");
      return;
    }
    const dest = await openFilePicker("export");
    if (!dest) return;
    const r = await fetch("/api/fs/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ src, dest }),
    });
    const d = await r.json();
    if (d.status === "success") pushToast("success", `Exported to ${d.path}`);
    else pushToast("error", d.error ?? "export failed");
  };

  // Opens the file picker, then saves the current graph to the chosen path.
  const saveAs = async (format: "ascii" | "binary") => {
    const current = useGraphStore.getState().activeFileName ?? "untitled";
    const defName = current + (format === "binary" ? ".ric" : ".riko");
    const p = await openFilePicker("save", { defaultName: defName });
    if (!p) return;
    await useGraphStore.getState().saveToPath(p, format);
  };

  // Exports the graph as a multi-file project into the chosen directory.
  const saveProject = async () => {
    const current = useGraphStore.getState().activeFileName ?? "graph_project";
    const p = await openFilePicker("save", { defaultName: current });
    if (!p) return;
    await useGraphStore.getState().exportProjectTo(p);
  };

  // Creates a new project folder (PyCharm-style preset tree) and opens it.
  const newProject = async () => {
    const p = await openFilePicker("save", { defaultName: "my_project" });
    if (!p) return;
    const r = await fetch("/api/project/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: p }),
    });
    const d = await r.json();
    if (d.status === "success") {
      useGraphStore.getState().bumpProject();
      pushToast("success", `Created project ${d.root}`);
    } else {
      pushToast("error", d.error ?? "new project failed");
    }
  };

  // Declarative menu tree: top-level labels map to flat or nested items.
  const menus: Menu[] = [
    {
      label: "File",
      items: [
        { label: "New Workflow", onClick: () => newWorkflow("Untitled") },
        { label: "New Project…", onClick: newProject },
        { label: "Save .riko…", onClick: () => saveAs("ascii"), disabled: busy },
        { label: "Save .ric…", onClick: () => saveAs("binary"), disabled: busy },
        {
          label: "Import / Export",
          submenu: [
            { label: "Import…", onClick: importFile },
            { label: "Export…", onClick: exportFile },
            { label: "Import Working Folder…", onClick: () => useGraphStore.getState().setImportFolderOpen(true) },
          ],
        },
        {
          label: "Export Code",
          submenu: [
            { label: "Export .py (PyTorch)", onClick: exportPython, disabled: busy },
            { label: "Export .py (Keras/TF)", onClick: exportKeras, disabled: busy },
            { label: "Export Project… (multi-file)", onClick: saveProject, disabled: busy },
          ],
        },
        { label: "Preferences…", onClick: () => useGraphStore.getState().setPrefsOpen(true) },
      ],
    },
    {
      label: "Run",
      items: [
        { label: "Inference", onClick: run, disabled: busy },
        { label: "Train (20 steps)", onClick: () => train(20, 1e-3), disabled: busy },
        { label: "Train (100 steps)", onClick: () => train(100, 1e-3), disabled: busy },
      ],
    },
    {
      label: "Data",
      items: [
        { label: "MNIST Loader", onClick: () => addNode("mnist_loader", { x: 200, y: 150 }) },
        { label: "CIFAR10 Loader", onClick: () => addNode("cifar10_loader", { x: 200, y: 150 }) },
        { label: "CSV Loader", onClick: () => addNode("csv_loader", { x: 200, y: 150 }) },
        { label: "Image Folder Loader", onClick: () => addNode("image_folder_loader", { x: 200, y: 150 }) },
        { label: "Tensor File Loader", onClick: () => addNode("tensor_file_loader", { x: 200, y: 150 }) },
        { label: "Generic Data Loader", onClick: () => addNode("dataloader", { x: 200, y: 150 }) },
      ],
    },
    {
      label: "View",
      items: [
        { label: `Theme: Light ${theme === "light" ? "✓" : ""}`, onClick: () => setTheme("light") },
        { label: `Theme: Dark ${theme === "dark" ? "✓" : ""}`, onClick: () => setTheme("dark") },
        { label: `Theme: System ${theme === "system" ? "✓" : ""}`, onClick: () => setTheme("system") },
        { label: `Theme: Liquid Glass ${theme === "glass" ? "✓" : ""}`, onClick: () => setTheme("glass") },
      ],
    },
    {
      label: "Help",
      items: [
        { label: "Welcome Screen", onClick: () => useGraphStore.getState().setWelcomeOpen(true) },
        { label: "About Entropia Riko", onClick: () => useGraphStore.getState().setAboutOpen(true) },
      ],
    },
  ];

  // The logo menu, dropdowns, workspace tabs, hidden file input, and status pill.
  return (
    <div className="menubar" onClick={(e) => e.stopPropagation()}>
      <div
        className={`menu-item app-logo-item ${openMenu === "__app__" ? "active" : ""}`}
        onClick={() => setOpenMenu(openMenu === "__app__" ? null : "__app__")}
        title="Entropia Riko"
      >
        <img className="menubar-logo" src="/brand/logo.svg" alt="Entropia Riko" />
        {openMenu === "__app__" && (
          <div className="dropdown app-menu" onClick={(e) => e.stopPropagation()}>
            <div
              className="dropdown-item"
              onClick={() => {
                useGraphStore.getState().setAboutOpen(true);
                setOpenMenu(null);
              }}
            >
              About Entropia Riko
            </div>
            <div
              className="dropdown-item"
              onClick={() => {
                useGraphStore.getState().setWelcomeOpen(true);
                setOpenMenu(null);
              }}
            >
              Welcome Screen
            </div>
            <div
              className="dropdown-item"
              onClick={() => {
                useGraphStore.getState().setPrefsOpen(true);
                setOpenMenu(null);
              }}
            >
              Preferences…
            </div>
          </div>
        )}
      </div>
      {menus.map((m) => (
        <div
          key={m.label}
          className={`menu-item ${openMenu === m.label ? "active" : ""}`}
          onClick={() => setOpenMenu(openMenu === m.label ? null : m.label)}
        >
          {m.label}
          {openMenu === m.label && (
            <div className="dropdown" onClick={(e) => e.stopPropagation()}>
              {m.items.map((item, i) =>
                item.submenu ? (
                  <div key={i} className="dropdown-item has-submenu">
                    <span className="submenu-label">{item.label} ▸</span>
                    <div className="submenu">
                      {item.submenu.map((sub, j) => (
                        <div
                          key={j}
                          className={`dropdown-item ${sub.disabled ? "disabled" : ""}`}
                          onClick={() => {
                            if (sub.disabled) return;
                            sub.onClick?.();
                            setOpenMenu(null);
                          }}
                        >
                          {sub.label}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div
                    key={i}
                    className={`dropdown-item ${item.disabled ? "disabled" : ""}`}
                    onClick={() => {
                      if (item.disabled) return;
                      item.onClick?.();
                      setOpenMenu(null);
                    }}
                  >
                    {item.label}
                  </div>
                )
              )}
            </div>
          )}
        </div>
      ))}
      <div className="menubar-sep" />
      <WorkspaceTabs
        workspaces={workspaces}
        activeId={activeWorkspaceId}
        onSwitch={onSwitchWorkspace}
        onAdd={onAddWorkspace}
        onRemove={onRemoveWorkspace}
        onRename={onRenameWorkspace}
        onDuplicate={onDuplicateWorkspace}
        onMove={onMoveWorkspace}
      />
      <input
        ref={fileRef}
        type="file"
        accept=".riko,.ric,.json,application/json,application/octet-stream"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) load(f);
          e.target.value = "";
        }}
      />
      <div className="spacer" />
      <span className={`status-pill ${status}`}>
        {status}
        {status !== "idle" && (
          <button
            className="status-pill-close"
            title="Dismiss"
            onClick={() => useGraphStore.setState({ status: "idle" })}
          >
            ✕
          </button>
        )}
      </span>
    </div>
  );
}
