# UI Standard

This document describes the visual language, layout model, and interaction
conventions of the Entropia Riko editor. It is the reference for keeping the UI
consistent as the app grows.

## Design language

The UI is a **frosted-glass, IDE-style** desktop editor — calm, compact, and
technical, not a marketing page. All colors, radii, and spacing come from CSS
design tokens in `src/ui/styles.css`, so components stay consistent across
themes.

Prefer:

- Clear hierarchy (title bar → menu bar → panels → content).
- Neutral, translucent surfaces and consistent 8px-based spacing.
- Compact technical panels with dense, legible controls.
- Useful status indicators (idle / running / success / error).
- Real labels on inputs and clear verbs on buttons.

Avoid:

- Landing-page hero sections and decorative complexity.
- Excessive gradients (the theme gradient is the only background).
- Vague labels ("stuff", "here", "click me").
- Hardcoded colors — use the `--color-*` / `--glass-*` tokens.

Typography: `--font` (Inter / system sans) for UI, `--font-mono` (JetBrains Mono
/ system mono) for code, ports, and logs.

---

## App shell layout

```text
Titlebar          [*] filename - Riko                    v0.1.0
Menu bar          (logo) File Run Data View Help | workspace tabs | status pill
Area tree         ┌──────────┬──────────────────────────────┐
                  │ Node Lib │  Canvas       │ Inspector     │
                  │ Assets   │               │ (or New File) │
                  ├──────────┴──────────────────────────────┤
                  │ Status / Logs        │ Loss Curve        │
                  └─────────────────────────────────────────┘
```

- **Title bar** (`Titlebar.tsx`): shows a `*` while the document is dirty, then
  `[filename] - Riko`, with the version pinned right (macOS window-title
  convention).
- **Menu bar** (`MenuBar.tsx`): app-logo menu + `File`, `Run`, `Data`, `View`,
  `Help` dropdowns, then the workspace tabs, then a right-aligned status pill.
- **Body**: the Blender-style *area tree* of resizable panels (below).

---

## Workspace panels (Blender-style areas)

The window layout is a **binary tree of splits** (`src/ui/areas.ts`). A split is
either a *row* (children side-by-side) or a *column* (children stacked); leaves
are *areas*, each showing one editor type. Every panel can therefore be split
both directions, resized, and merged away.

### Interaction

Each panel has a slim header bar and a bottom-right **corner grip**:

- **Drag up / left** → split the panel (a live preview line follows the mouse).
- **Drag down / right** → merge it into a sibling (the sibling is shaded and
  bold-bordered while targeting).
- Release to commit.

The header bar holds a **type dropdown** (switch this window to any editor type)
and a **✕** close button. The type dropdown is a **multi-column menu** grouped by
category — `Editor`, `Data`, `Tools` — not a flat list.

### Panel types

| Type        | Label          | Category | Contents                                       |
|-------------|----------------|----------|------------------------------------------------|
| `canvas`    | Graph          | Editor   | React Flow graph canvas (breadcrumb, frames).  |
| `inspector` | Inspector      | Editor   | Parameters, inputs/outputs, previews, debug.   |
| `status`    | Status / Logs  | Editor   | Run status and timestamped log lines.          |
| `loss`      | Loss Curve     | Editor   | Live loss chart (SSE-streamed training).       |
| `nodes`     | Node Library   | Data     | Searchable node registry.                      |
| `files`     | Asset Library  | Data     | Disk-backed `.riko`/`.ric` asset tree.         |
| `project`   | New File       | Data     | Working-directory mini file manager.           |
| `code`      | Code Editor    | Tools    | Notepad-style code editor.                     |
| `pad`       | Handwriting Pad| Tools    | 28×28 digit pad → `constant` node.             |
| `docs`      | Documentation  | Tools    | In-app docs.                                   |
| `plugins`   | Plugins        | Tools    | Plugin list and enable/disable toggles.        |

### Workspaces

Multiple named workspaces are available as **tabs** (`WorkspaceTabs.tsx`). Each
tab is an independent area-tree layout and can be added, renamed, duplicated,
reordered, and closed. Presets ship in `WORKSPACE_PRESETS` (`src/ui/areas.ts`):
Layout, Code, Inference, Training, Hyperparameter Tuning, MNIST Studio, Image
Classifier, Text → Image, Object Detection, Text Classifier, Text Gen, and
Embeddings/JEPA.

---

## Floating windows

All app-level dialogs are **draggable floating windows** rendered by
`FloatingWindow.tsx`: a title bar that can be dragged to reposition, a `✕` close
button, and a configurable width/z-index. The dialog list is:

- **About** (`AboutPanel.tsx`)
- **Preferences** (`PreferencesPanel.tsx`) — category sidebar (Appearance /
  Plugins / About) + settings pane, macOS/Blender style
- **Import Working Folder** (`ImportFolderPanel.tsx`)
- **File Picker** (`FilePicker.tsx`) — Windows-style import/export/save browser

The **Welcome screen** (`WelcomePanel.tsx`) is the one exception: it is a
full-screen overlay (not draggable), with a New File / Presets column and a
Recent Files column, plus Open and Recover Last Session actions.

Conventions for dialogs:

- Use `FloatingWindow` for anything detachable-feeling; keep one close action
  that is reachable without dragging.
- Modals open from the menu bar or a panel button and close on ✕, `Esc`, or a
  Cancel button.
- Single-purpose dialogs (file picker) may use a promise API
  (`openFilePicker("import" | "export" | "save")`).

---

## Themes

Four themes ship, selectable from **View → Theme** and
**Preferences → Appearance**:

| Mode      | Behavior                                                              |
|-----------|-----------------------------------------------------------------------|
| `light`   | Always the light palette.                                             |
| `dark`    | Always the dark palette.                                              |
| `system`  | Follows the OS `prefers-color-scheme`.                                |
| `glass`   | **Liquid Glass** — Apple-style extra-translucent frosted glass.       |

Implementation (`src/ui/theme.ts`): the mode is stored in `localStorage`
(`entropia_riko_theme`) and applied by setting `data-theme` on `<html>`; the CSS
tokens in `styles.css` switch accordingly. `system` leaves the attribute unset
so `prefers-color-scheme` drives the dark tokens. Liquid Glass raises
`--glass-blur` to 40px, `--glass-saturate` to 200%, lightens surfaces, and adds
specular inset highlights to the menu bar, panels, dropdowns, context menus, and
nodes.

A **background image** (optional cover URL) can be set in Preferences; it is
stored in `localStorage` (`entropia_riko_background`) and applied to the body.
Both theme and background image are persisted into the document's `settings`
on save (see `FILE_FORMAT.md`).

---

## Graph canvas conventions

The canvas is React Flow (`@xyflow/react`) rendered by `GraphCanvas.tsx`:

- **Right-click** opens a **searchable node menu** (`ContextMenu.tsx` +
  `PopupMenu.tsx`) that lists "＋ Add Frame" plus every registered node, grouped
  by category and filtered as you type.
- **Drag from an output handle to an input handle** connects ports; invalid or
  stale port names fall back gracefully at execution.
- A **zoom pill** (+, fit view, −) is docked bottom-right.
- Custom **node cards** (`NodeCard.tsx`) show the title, ports, and live output
  previews; results stream back from `/api/execute` with shape / dtype / device /
  summary.

### Subgraph navigation

- **Double-click** a `graph_reference` or `import` node to enter its graph.
- A **breadcrumb** in the graph's top-left (`root / subgraph`) shows the current
  level; click an earlier crumb to exit back up. Stack state lives in
  `graphStack` (`graphStore.ts`).

### Node frames

**Frames** are visual grouping boxes (`graph-frame` in `GraphCanvas.tsx`),
rendered inside the viewport via `ViewportPortal`:

- Add via **right-click → Add Frame**.
- Drag the title bar to move (zoom-corrected); double-click to rename; ✕ removes.
- Frames are stored in the `frames` array of `graphStore.ts` (id, title,
  x/y/width/height) and are not part of the serialized document.

---

## Panels & tools conventions

- **Code editor** (`CodeEditor.tsx`) is Notepad-style: `File` / `Edit` dropdown
  menus plus a quick-action toolbar (New, Open, Save, Undo/Redo, Cut/Copy/Paste,
  Select All). Exported PyTorch code is loaded into it via
  "Generate from Graph" or a file's "Preview PyTorch Code".
- **Handwriting pad** (`HandwritingPad.tsx`) draws a 28×28 digit and sends it as
  a `constant` node with value shape `[1, 1, 28, 28]` to feed the MNIST example.
- **Plugin panel** (`PluginPanel.tsx`) lists loaded plugins with enable/disable
  toggles; it is reused both as a workspace panel and inside Preferences.
- **File managers** — two complementary tools:
  - **Asset Library** (`FileManager.tsx`): a searchable tree of the built-in
    `workflows/` + `examples/` assets. Per file: open, **⤢ expand full nodes**
    (inline the graph instead of a subgraph reference), **⤡ preview PyTorch
    code**, and a save-current action. Binary `.ric` files get a `BIN` badge.
  - **New File** (`ProjectPanel.tsx`): an IDE-explorer tree of the *working
    folder* with a right-click menu (New File / New Folder / Open / Preview
    PyTorch Code / Import as Node / Rename / Delete) and drag-and-drop moving.
- **Windows-style file picker** (`FilePicker.tsx`): used for every import /
  export / save. It has Quick access (Home, Desktop, Documents, Downloads,
  Working Folder) and Recent sidebars, back/forward/refresh navigation, a path
  field, a file list, and a File-name field. Import/export *copy* files/folders
  via the backend rather than triggering browser downloads.

---

## Status & feedback

- **Status pill** (menu bar, right-aligned) shows `idle` / `running` / `success`
  / `error` and can be dismissed back to idle.
- **Status / Logs panel** shows timestamped log lines (capped at 80).
- **Toasts** (`ToastStack.tsx`) surface success/error notifications (non-error
  toasts auto-dismiss after 6s; errors persist until dismissed).
- **Loss Curve** panel streams per-step loss over SSE and plots it in an SVG.
