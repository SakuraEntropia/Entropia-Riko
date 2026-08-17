/** "New File" mini file manager — a working-directory tree with a right-click
 * context menu (create / folder / rename / delete / open / preview code /
 * import-as-node). IDE-explorer style, above the inspector. */
import { useEffect, useState } from "react";
import { useGraphStore } from "../store/graphStore";
import { PopupMenu, type MenuEntry } from "./PopupMenu";

interface TreeNode {
  name: string;
  path: string;
  type: "dir" | "file";
  children?: TreeNode[];
}

interface MenuState {
  x: number;
  y: number;
  target: TreeNode | null; // null → tree background
}

export function ProjectPanel() {
  const pushToast = useGraphStore((s) => s.pushToast);
  const openProjectFile = useGraphStore((s) => s.openProjectFile);
  const importModule = useGraphStore((s) => s.importModule);
  const projectVersion = useGraphStore((s) => s.projectVersion);
  const [tree, setTree] = useState<TreeNode[]>([]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [menu, setMenu] = useState<MenuState | null>(null);
  const [dragPath, setDragPath] = useState<string | null>(null);

  // (Re)load the working-directory tree from the API server.
  const refresh = async () => {
    try {
      const r = await fetch("/api/project/tree");
      const d = await r.json();
      setTree(Array.isArray(d.tree) ? d.tree : []);
    } catch {
      setTree([]);
    }
  };

  useEffect(() => {
    refresh();
  }, [projectVersion]);

  // Thin JSON POST helper shared by every project mutation endpoint.
  const api = async (url: string, body: Record<string, unknown>) => {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return r.json();
  };

  const newFile = async (dir: string) => {
    const name = window.prompt("New file name", "untitled.riko");
    if (!name) return;
    const d = await api("/api/project/create", { name, dir });
    if (d.status === "success") {
      await refresh();
      openProjectFile(d.path);
    } else {
      pushToast("error", d.error ?? "create failed");
    }
  };

  const newFolder = async (dir: string) => {
    const name = window.prompt("New folder name");
    if (!name) return;
    const d = await api("/api/project/mkdir", { name, dir });
    if (d.status === "success") {
      setExpanded((e) => ({ ...e, [dir]: true }));
      await refresh();
    } else {
      pushToast("error", d.error ?? "mkdir failed");
    }
  };

  const rename = async (node: TreeNode) => {
    const newName = window.prompt("Rename", node.name);
    if (!newName) return;
    const d = await api("/api/project/rename", { path: node.path, newName });
    if (d.status === "success") await refresh();
    else pushToast("error", d.error ?? "rename failed");
  };

  const remove = async (node: TreeNode) => {
    if (!window.confirm(`Delete ${node.path}?`)) return;
    const d = await api("/api/project/delete", { path: node.path });
    if (d.status === "success") {
      await refresh();
      pushToast("success", `Deleted ${node.name}`);
    } else {
      pushToast("error", d.error ?? "delete failed");
    }
  };

  const previewCode = async (node: TreeNode) => {
    const d = await api("/api/project/code", { path: node.path });
    if (d.status === "success") {
      useGraphStore.getState().setCodeBuffer(d.code ?? "");
      pushToast("success", "Code → Code Editor panel");
    } else {
      pushToast("error", d.error ?? "code generation failed");
    }
  };

  const move = async (path: string, targetDir: string) => {
    const d = await api("/api/project/move", { path, targetDir });
    if (d.status === "success") {
      await refresh();
      pushToast("success", `Moved ${path} → ${targetDir || "(root)"}`);
    } else {
      pushToast("error", d.error ?? "move failed");
    }
  };

  // The right-click menu differs by target: a file, a directory, or the tree
  // background (menu.target === null).
  const menuEntries = (): MenuEntry[] => {
    const t = menu?.target ?? null;
    if (t && t.type === "file") {
      return [
        { id: "open", label: "Open", onSelect: () => openProjectFile(t.path) },
        { id: "code", label: "Preview PyTorch Code", description: "→ Code Editor panel", onSelect: () => previewCode(t) },
        { id: "import", label: "Import as Node", description: "reference this graph", onSelect: () => importModule(t.name.replace(/\.(riko|ric)$/i, ""), { x: 260, y: 150 }) },
        { id: "sep1", label: "", separator: true },
        { id: "rename", label: "Rename", onSelect: () => rename(t) },
        { id: "delete", label: "Delete", danger: true, onSelect: () => remove(t) },
      ];
    }
    if (t && t.type === "dir") {
      return [
        { id: "newfile", label: "New File", onSelect: () => newFile(t.path) },
        { id: "newfolder", label: "New Folder", onSelect: () => newFolder(t.path) },
        { id: "sep1", label: "", separator: true },
        { id: "rename", label: "Rename", onSelect: () => rename(t) },
        { id: "delete", label: "Delete", danger: true, onSelect: () => remove(t) },
      ];
    }
    return [
      { id: "newfile", label: "New File", onSelect: () => newFile("") },
      { id: "newfolder", label: "New Folder", onSelect: () => newFolder("") },
      { id: "sep1", label: "", separator: true },
      { id: "refresh", label: "Refresh", onSelect: refresh },
    ];
  };

  // Recursively render the tree, indenting each level by depth.
  const renderNode = (node: TreeNode, depth: number) => {
    if (node.type === "dir") {
      const open = !!expanded[node.path];
      return (
        <div key={node.path}>
          <div
            className="proj-row"
            style={{ paddingLeft: 8 + depth * 12 }}
            draggable
            onDragStart={(e) => {
              setDragPath(node.path);
              e.dataTransfer.effectAllowed = "move";
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.dataTransfer.dropEffect = "move";
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();
              // Move the dragged item into this directory (skip dropping onto itself).
              if (dragPath && dragPath !== node.path) move(dragPath, node.path);
              setDragPath(null);
            }}
            onClick={() => setExpanded((e) => ({ ...e, [node.path]: !open }))}
            onContextMenu={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setMenu({ x: e.clientX, y: e.clientY, target: node });
            }}
          >
            <span className="proj-caret">{open ? "▾" : "▸"}</span>
            <span className="proj-icon">▦</span>
            <span className="proj-name">{node.name}</span>
          </div>
          {open && (node.children ?? []).map((c) => renderNode(c, depth + 1))}
        </div>
      );
    }
    return (
      <div
        key={node.path}
        className="proj-row proj-file"
        style={{ paddingLeft: 8 + depth * 12 }}
        draggable
        onDragStart={(e) => {
          setDragPath(node.path);
          e.dataTransfer.effectAllowed = "move";
        }}
        onClick={() => openProjectFile(node.path)}
        onContextMenu={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setMenu({ x: e.clientX, y: e.clientY, target: node });
        }}
      >
        <span className="proj-caret">·</span>
        <span className="proj-icon">▤</span>
        <span className="proj-name">{node.name}</span>
      </div>
    );
  };

  return (
    <div className="project-panel">
      <div className="panel-head">
        <span className="panel-head-title">New File</span>
        <span className="spacer" />
        <button className="panel-btn" title="Refresh" onClick={refresh}>
          ↻
        </button>
      </div>
      <div
        className="proj-tree"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          // Dropping on empty tree area moves the dragged item to the project root.
          if (dragPath) move(dragPath, "");
          setDragPath(null);
        }}
        onContextMenu={(e) => {
          e.preventDefault();
          setMenu({ x: e.clientX, y: e.clientY, target: null });
        }}
      >
        {tree.map((n) => renderNode(n, 0))}
        {tree.length === 0 && (
          <div className="welcome-empty">Right-click to create a file.</div>
        )}
      </div>
      {menu && (
        <PopupMenu entries={menuEntries()} x={menu.x} y={menu.y} onClose={() => setMenu(null)} />
      )}
    </div>
  );
}
