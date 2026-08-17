/** Graph editor state (zustand). Owns React Flow nodes/edges, selection,
 * run status, logs, node definitions, and workflow save/load. */
import { create } from "zustand";
import {
  type Node,
  type Edge,
  type Connection,
  type OnNodesChange,
  type OnEdgesChange,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from "@xyflow/react";
import {
  NODE_DEFINITIONS,
  type NodeDef,
  type PortDef,
  type ParamDef,
} from "../nodes/nodeDefinitions";
import { APP_VERSION } from "../version";

export interface ThnNodeData extends Record<string, unknown> {
  type: string;
  label: string;
  category: string;
  params: Record<string, unknown>;
  inputs: PortDef[];
  outputs: PortDef[];
  hasError: boolean;
  errorMsg?: string;
}

export interface TensorPreview {
  summary: string;
  shape: number[];
  dtype: string;
  device: string;
  dataPreview: string;
  dataKind?: string;
  image?: string;
}

type Status = "idle" | "running" | "success" | "error";

export interface Toast {
  id: number;
  kind: "info" | "success" | "error";
  message: string;
}

let toastUid = 0;

export interface WorkflowDef {
  id: string;
  name: string;
  nodes: Node<ThnNodeData>[];
  edges: Edge[];
}

export interface FileImportRef {
  spec: string;
  path: string | null;
  resolved: boolean;
}

export interface FileInfo {
  name: string;
  path: string;
  imports: FileImportRef[];
  format?: "ascii" | "binary";
}

export interface Frame {
  id: string;
  title: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface GraphLevel {
  name: string;
  path: string;
  nodes: Node<ThnNodeData>[];
  edges: Edge[];
}

interface ApiNodeDef {
  type: string;
  label: string;
  category: string;
  inputs: { name: string; data_kind: string; required: boolean }[];
  outputs: { name: string; data_kind: string }[];
  parameters: {
    name: string;
    kind: string;
    default: unknown;
    required: boolean;
    dtype: string | null;
    browse?: string | null;
  }[];
}

function labelize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Converts an API node schema into the editor's NodeDef shape, labelizing
 * port/parameter names and casting parameter kinds. */
export function apiNodeToDef(n: ApiNodeDef): NodeDef {
  return {
    type: n.type,
    label: n.label,
    category: n.category,
    inputs: n.inputs.map((i) => ({
      name: i.name,
      label: labelize(i.name),
      dataKind: i.data_kind,
      direction: "in" as const,
    })),
    outputs: n.outputs.map((o) => ({
      name: o.name,
      label: labelize(o.name),
      dataKind: o.data_kind,
      direction: "out" as const,
    })),
    parameters: n.parameters.map((p): ParamDef => ({
      name: p.name,
      label: labelize(p.name),
      kind: p.kind as "scalar" | "any" | "path",
      default: p.default,
      required: p.required,
      dtype: p.dtype ?? undefined,
      browse: (p.browse as "open" | "save") ?? undefined,
    })),
  };
}

interface GraphState {
  nodeDefs: NodeDef[];
  nodes: Node<ThnNodeData>[];
  edges: Edge[];
  frames: Frame[];
  addFrame: (title: string, position: { x: number; y: number }) => void;
  updateFrame: (id: string, partial: Partial<Frame>) => void;
  removeFrame: (id: string) => void;
  graphStack: GraphLevel[];
  enterSubgraph: (nodeId: string) => Promise<void>;
  exitToLevel: (index: number) => void;
  selectedNodeId: string | null;
  status: Status;
  logs: string[];
  results: Record<string, Record<string, TensorPreview>>;
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: (conn: Connection) => void;
  addNode: (type: string, position: { x: number; y: number }) => void;
  onSelectNode: (id: string | null) => void;
  updateParam: (nodeId: string, name: string, value: unknown) => void;
  removeNode: (nodeId: string) => void;
  run: () => Promise<void>;
  loadNodeDefs: () => Promise<void>;
  exportPython: () => Promise<void>;
  exportKeras: () => Promise<void>;
  workflows: WorkflowDef[];
  activeWorkflowId: string | null;
  newWorkflow: (name: string) => void;
  switchWorkflow: (id: string) => void;
  save: () => void;
  exportRiko: () => void;
  exportBinary: () => Promise<void>;
  exportProjectTo: (dir: string) => Promise<void>;
  generatePythonCode: () => Promise<void>;
  previewAssetCode: (path: string) => Promise<void>;
  load: (file: File) => Promise<void>;
  log: (msg: string) => void;
  fileList: FileInfo[];
  activeFileName: string | null;
  activeFilePath: string | null;
  dirty: boolean;
  refreshFiles: () => Promise<void>;
  openFile: (path: string) => Promise<void>;
  openProjectFile: (path: string) => Promise<void>;
  importFileInline: (path: string) => Promise<void>;
  saveFileToDisk: (name: string, format?: "ascii" | "binary") => Promise<void>;
  saveToPath: (path: string, format?: "ascii" | "binary") => Promise<void>;
  importModule: (spec: string, position: { x: number; y: number }) => void;
  losses: number[];
  train: (steps?: number, lr?: number) => Promise<void>;
  recentFiles: string[];
  newBlank: () => void;
  welcomeOpen: boolean;
  setWelcomeOpen: (v: boolean) => void;
  prefsOpen: boolean;
  setPrefsOpen: (v: boolean) => void;
  aboutOpen: boolean;
  setAboutOpen: (v: boolean) => void;
  importFolderOpen: boolean;
  setImportFolderOpen: (v: boolean) => void;
  filePickerOpen: boolean;
  filePickerMode: "import" | "export" | "save";
  setFilePicker: (open: boolean, mode: "import" | "export" | "save") => void;
  projectVersion: number;
  bumpProject: () => void;
  codeBuffer: string;
  setCodeBuffer: (v: string) => void;
  toasts: Toast[];
  pushToast: (kind: Toast["kind"], message: string) => void;
  dismissToast: (id: number) => void;
}

let idCounter = 0;
// Generates unique ids scoped by a prefix (e.g. "edge", "wf", "n").
function nextId(prefix: string): string {
  idCounter += 1;
  return `${prefix}_${idCounter}`;
}

/** Assembles the serializable `.riko` document from the current graph,
 * pulling name/theme/background from the active file and localStorage. */
function buildDoc(nodes: Node<ThnNodeData>[], edges: Edge[]) {
  let name = "untitled";
  let theme = "";
  let backgroundImage = "";
  try {
    name = useGraphStore.getState().activeFileName ?? "untitled";
    theme = localStorage.getItem("entropia_riko_theme") ?? "";
    backgroundImage = localStorage.getItem("entropia_riko_background") ?? "";
  } catch {
    /* ignore */
  }
  return {
    version: "1.0",
    metadata: { name, app: "entropia-riko", appVersion: APP_VERSION },
    nodes: nodes.map((n) => ({
      id: n.id,
      type_name: n.data.type,
      label: n.data.label,
      category: n.data.category,
      position: [n.position.x, n.position.y],
      parameters: n.data.params,
      inputs: [],
      outputs: [],
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source_node: e.source,
      source_port: e.sourceHandle,
      target_node: e.target,
      target_port: e.targetHandle,
    })),
    settings: { theme, backgroundImage },
  };
}

interface DocNode {
  id: string;
  type_name: string;
  label?: string;
  category?: string;
  position?: number[];
  parameters?: Record<string, unknown>;
}

interface DocEdge {
  id: string;
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
}

/** Converts a saved `.riko` doc back into React Flow nodes/edges, re-attaching
 * port/parameter definitions from the node registry. */
function docToReactFlow(
  doc: { nodes?: DocNode[]; edges?: DocEdge[] },
  nodeDefs: NodeDef[]
) {
  const nodes: Node<ThnNodeData>[] = (doc.nodes || []).map((n) => {
    const def = nodeDefs.find((d) => d.type === n.type_name);
    return {
      id: n.id,
      type: "thnNode",
      position: { x: n.position?.[0] ?? 0, y: n.position?.[1] ?? 0 },
      data: {
        type: n.type_name,
        label: n.label || def?.label || n.type_name,
        category: n.category || def?.category || "",
        params: n.parameters || {},
        inputs: def?.inputs || [],
        outputs: def?.outputs || [],
        hasError: false,
      },
    };
  });
  const edges: Edge[] = (doc.edges || []).map((e) => ({
    id: e.id,
    source: e.source_node,
    target: e.target_node,
    sourceHandle: e.source_port,
    targetHandle: e.target_port,
  }));
  return { nodes, edges };
}

/** Resolves a subgraph `file`/`module` spec to a real server path by trying
 * several candidate locations in order. */
async function resolveSubgraphFile(spec: string): Promise<string | null> {
  const s = (spec || "").trim();
  if (!s) return null;
  const candidates: string[] = [];
  if (s.includes("/") || s.endsWith(".riko") || s.endsWith(".ric")) candidates.push(s);
  candidates.push(`examples/models/${s}.riko`);
  candidates.push(`examples/${s}.riko`);
  candidates.push(`workflows/${s}.riko`);
  candidates.push(`${s}.riko`);
  for (const c of candidates) {
    try {
      const resp = await fetch(`/api/files/content?path=${encodeURIComponent(c)}`);
      const data = await resp.json();
      if (data.status === "success") return c;
    } catch {
      /* try next */
    }
  }
  return null;
}

const RECENT_KEY = "entropia_riko_recent";

/** Reads the recent-files list from localStorage, tolerating corrupt data. */
function loadRecent(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    const arr = raw ? (JSON.parse(raw) as unknown) : [];
    return Array.isArray(arr) ? (arr as string[]).filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

/** Persists the recent-files list (capped at 10 entries). */
function saveRecent(paths: string[]) {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(paths.slice(0, 10)));
  } catch {
    /* ignore */
  }
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodeDefs: NODE_DEFINITIONS,
  workflows: [],
  activeWorkflowId: null,
  nodes: [],
  edges: [],
  frames: [],
  graphStack: [{ name: "root", path: "", nodes: [], edges: [] }],
  selectedNodeId: null,
  status: "idle",
  logs: [],
  results: {},
  fileList: [],
  activeFileName: null,
  activeFilePath: null,
  dirty: false,
  losses: [],
  recentFiles: loadRecent(),
  welcomeOpen: true,
  prefsOpen: false,
  aboutOpen: false,
  importFolderOpen: false,
  filePickerOpen: false,
  filePickerMode: "import",
  projectVersion: 0,
  codeBuffer: "",
  toasts: [],

  onNodesChange: (changes) => {
    const nodes = applyNodeChanges(changes, get().nodes) as Node<ThnNodeData>[];
    // Selection changes are not content edits — ignore them for the dirty flag.
    const meaningful = changes.some((c) => c.type !== "select");
    set(meaningful ? { nodes, dirty: true } : { nodes });
  },
  onEdgesChange: (changes) => {
    const edges = applyEdgeChanges(changes, get().edges);
    const meaningful = changes.some((c) => c.type !== "select");
    set(meaningful ? { edges, dirty: true } : { edges });
  },

  onConnect: (conn) => {
    if (!conn.source || !conn.target) return;
    // Create the edge with a fresh id and mark the graph dirty.
    const id = nextId("edge");
    set({ edges: addEdge({ ...conn, id }, get().edges), dirty: true });
    get().log(`连接 ${conn.source}.${conn.sourceHandle} → ${conn.target}.${conn.targetHandle}`);
  },

  addNode: (type, position) => {
    const def = get().nodeDefs.find((n) => n.type === type);
    const id = nextId(type);
    // Seed params with defaults from the node definition.
    const params: Record<string, unknown> = {};
    if (def) {
      for (const p of def.parameters) params[p.name] = p.default;
    }
    const node: Node<ThnNodeData> = {
      id,
      type: "thnNode",
      position,
      data: {
        type,
        label: def?.label ?? type,
        category: def?.category ?? "Other",
        params,
        inputs: def?.inputs ?? [],
        outputs: def?.outputs ?? [{ name: "result", label: "Result", dataKind: "tensor", direction: "out" as const }],
        hasError: false,
      },
    };
    set({ nodes: [...get().nodes, node], selectedNodeId: id, dirty: true });
    get().log(`添加节点 ${def?.label ?? type} (${id})`);
  },

  onSelectNode: (id) => set({ selectedNodeId: id }),

  updateParam: (nodeId, name, value) =>
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, params: { ...n.data.params, [name]: value } } }
          : n
      ),
      dirty: true,
    }),

  removeNode: (nodeId) =>
    set({
      nodes: get().nodes.filter((n) => n.id !== nodeId),
      edges: get().edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
      selectedNodeId: get().selectedNodeId === nodeId ? null : get().selectedNodeId,
      dirty: true,
    }),

  addFrame: (title, position) => {
    const id = nextId("frame");
    set({
      frames: [
        ...get().frames,
        { id, title: title.trim() || "Frame", x: position.x, y: position.y, width: 360, height: 220 },
      ],
      dirty: true,
    });
  },

  updateFrame: (id, partial) =>
    set({
      frames: get().frames.map((f) => (f.id === id ? { ...f, ...partial } : f)),
      dirty: true,
    }),

  removeFrame: (id) =>
    set({ frames: get().frames.filter((f) => f.id !== id), dirty: true }),

  enterSubgraph: async (nodeId) => {
    const node = get().nodes.find((n) => n.id === nodeId);
    if (!node) return;
    const spec = (node.data.params.file as string) ?? (node.data.params.module as string);
    if (!spec) {
      get().pushToast("error", "该节点没有 file/module 参数");
      return;
    }
    const path = await resolveSubgraphFile(String(spec));
    if (!path) {
      get().pushToast("error", `找不到子图: ${spec}`);
      return;
    }
    try {
      const resp = await fetch(`/api/files/content?path=${encodeURIComponent(path)}`);
      const data = await resp.json();
      if (data.status !== "success") {
        get().pushToast("error", data.error ?? "打开子图失败");
        return;
      }
      const { nodes, edges } = docToReactFlow(data.doc, get().nodeDefs);
      // Snapshot the current level onto the stack before pushing the child.
      const stack = [...get().graphStack];
      const cur = stack[stack.length - 1];
      stack[stack.length - 1] = { ...cur, nodes: get().nodes, edges: get().edges };
      const name = path.split("/").pop()?.replace(/\.(riko|ric)$/i, "") ?? path;
      stack.push({ name, path, nodes, edges });
      set({
        graphStack: stack,
        nodes,
        edges,
        selectedNodeId: null,
        results: {},
        status: "idle",
        dirty: false,
      });
      get().log(`进入子图 ${path}`);
    } catch (e) {
      get().pushToast("error", e instanceof Error ? e.message : String(e));
    }
  },

  exitToLevel: (index) => {
    // Truncate the stack to the target level and restore its saved graph.
    const stack = get().graphStack.slice(0, index + 1);
    const level = stack[index];
    if (!level) return;
    set({
      graphStack: stack,
      nodes: level.nodes,
      edges: level.edges,
      selectedNodeId: null,
      results: {},
      status: "idle",
      dirty: false,
    });
    get().log(`返回层级 ${level.name}`);
  },

  run: async () => {
    set({ status: "running" });
    get().log("开始执行图（调用 Python runtime）...");
    try {
      // POST the serialized doc to the Python runtime and map outputs per node/port.
      const doc = buildDoc(get().nodes, get().edges);
      const resp = await fetch("/api/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(doc),
      });
      const data = await resp.json();
      if (data.status === "success") {
        const results: Record<string, Record<string, TensorPreview>> = {};
        const outputs = data.outputs as Record<string, Record<string, {
          shape: number[]; dtype: string; device: string; summary: string;
          data: unknown; data_kind?: string; image?: string;
        }>>;
        for (const [nid, ports] of Object.entries(outputs)) {
          results[nid] = {};
          for (const [port, tv] of Object.entries(ports)) {
            results[nid][port] = {
              summary: tv.summary,
              shape: tv.shape,
              dtype: tv.dtype,
              device: tv.device,
              dataPreview:
                tv.data === null || tv.data === undefined
                  ? ""
                  : JSON.stringify(tv.data).slice(0, 120),
              dataKind: tv.data_kind,
              image: tv.image,
            };
          }
        }
        set({
          results,
          status: "success",
          nodes: get().nodes.map((n) => ({
            ...n,
            data: { ...n.data, hasError: false, errorMsg: undefined },
          })),
        });
        get().log("执行成功");
      } else {
        set({ status: "error" });
        for (const e of data.errors as string[]) {
          get().log(`错误: ${e}`);
          get().pushToast("error", e);
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      set({ status: "error" });
      get().log(`请求失败: ${msg}（请确认 API server 已启动）`);
      get().pushToast("error", `请求失败: ${msg}`);
    }
  },

  loadNodeDefs: async () => {
    try {
      // Fetch the live node registry from the API; fall back to the bundled list.
      const resp = await fetch("/api/nodes");
      const data = await resp.json();
      const defs = (data.nodes as ApiNodeDef[]).map(apiNodeToDef);
      if (defs.length > 0) {
        set({ nodeDefs: defs });
        get().log(`从 API 加载 ${defs.length} 个节点定义`);
      }
    } catch (e) {
      get().log(`API 节点加载失败: ${e instanceof Error ? e.message : String(e)}。请启动 API server: uvicorn src.server.app:app --port 8000`);
    }
  },

  save: () => {
    // Download the current graph as a local `workflow.riko` JSON file.
    const doc = buildDoc(get().nodes, get().edges);
    const blob = new Blob([JSON.stringify(doc, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "workflow.riko";
    a.click();
    URL.revokeObjectURL(url);
    set({ dirty: false });
    get().log("已保存工作流 (workflow.riko)");
  },

  exportRiko: () => {
    // Download the graph as a `.riko` file named after the active file.
    const doc = buildDoc(get().nodes, get().edges);
    const name = (get().activeFileName ?? "workflow").replace(/\.(riko|ric)$/i, "");
    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.riko`;
    a.click();
    URL.revokeObjectURL(url);
    set({ dirty: false });
    get().log(`已导出 ${name}.riko`);
  },

  exportBinary: async () => {
    try {
      // Ask the server to serialize the doc to binary, then download the bytes.
      const doc = buildDoc(get().nodes, get().edges);
      const resp = await fetch("/api/export_binary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(doc),
      });
      const data = await resp.json();
      if (data.status !== "success") throw new Error(data.error);
      const name = (get().activeFileName ?? "workflow").replace(/\.(riko|ric)$/i, "");
      const bytes = Uint8Array.from(atob(data.base64), (c) => c.charCodeAt(0));
      const blob = new Blob([bytes], { type: "application/octet-stream" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${name}.ric`;
      a.click();
      URL.revokeObjectURL(url);
      set({ dirty: false });
      get().log(`已导出 ${name}.ric`);
    } catch (e) {
      get().log(`导出 .ric 失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  },

  exportProjectTo: async (dir) => {
    // Send the doc + destination directory to the server to emit a multi-file project.
    const doc = buildDoc(get().nodes, get().edges);
    try {
      const resp = await fetch("/api/export_project", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc, dir }),
      });
      const data = await resp.json();
      if (data.status === "success") {
        get().pushToast("success", `Exported project → ${data.root}`);
        get().log(`已导出多文件 PyTorch 工程: ${(data.files as string[]).join(", ")}`);
      } else {
        get().pushToast("error", data.error ?? "导出工程失败");
      }
    } catch (e) {
      get().pushToast("error", e instanceof Error ? e.message : String(e));
    }
  },

  load: async (file: File) => {
    try {
      // Parse the file: `.ric` is binary (server decode), everything else is JSON text.
      let doc: Record<string, unknown>;
      if (file.name.toLowerCase().endsWith(".ric")) {
        // binary format → decode via the server
        const buf = await file.arrayBuffer();
        const resp = await fetch("/api/files/decode", {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: buf,
        });
        const data = await resp.json();
        if (data.status !== "success") throw new Error(data.error);
        doc = data.doc;
      } else {
        const text = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.onerror = () => reject(reader.error);
          reader.readAsText(file);
        });
        doc = JSON.parse(text);
      }
      const { nodes, edges } = docToReactFlow(doc as never, get().nodeDefs);
      set({ nodes, edges, frames: [], graphStack: [{ name: file.name.replace(/\.(riko|ric)$/i, "") || "workflow", path: file.name, nodes, edges: [] }], selectedNodeId: null, results: {}, status: "idle", dirty: false });
      get().log(`已加载工作流（${nodes.length} 节点, ${edges.length} 连接）`);
    } catch (e) {
      get().log(`加载失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  },

  exportPython: async () => {
    // Download server-generated PyTorch source as `workflow.py`.
    const doc = buildDoc(get().nodes, get().edges);
    try {
      const resp = await fetch("/api/export_python", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(doc),
      });
      const data = await resp.json();
      if (data.status === "success") {
        const blob = new Blob([data.code as string], { type: "text/x-python" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "workflow.py";
        a.click();
        URL.revokeObjectURL(url);
        get().log("已导出 Python (workflow.py)");
      } else {
        get().log(`导出失败: ${data.error}`);
      }
    } catch (e) {
      get().log(`导出失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  },

  exportKeras: async () => {
    // Download server-generated Keras/TensorFlow source as `workflow_keras.py`.
    const doc = buildDoc(get().nodes, get().edges);
    try {
      const resp = await fetch("/api/export_keras", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(doc),
      });
      const data = await resp.json();
      if (data.status === "success") {
        const blob = new Blob([data.code as string], { type: "text/x-python" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "workflow_keras.py";
        a.click();
        URL.revokeObjectURL(url);
        get().log("已导出 Keras (workflow_keras.py)");
      } else {
        get().log(`导出失败: ${data.error}`);
      }
    } catch (e) {
      get().log(`导出失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  },

  generatePythonCode: async () => {
    // Generate PyTorch source and stage it in the code editor buffer.
    const doc = buildDoc(get().nodes, get().edges);
    try {
      const resp = await fetch("/api/export_python", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(doc),
      });
      const data = await resp.json();
      if (data.status === "success") {
        set({ codeBuffer: data.code as string });
        get().log("已生成 PyTorch 代码（代码编辑器窗口）");
      } else {
        get().pushToast("error", data.error ?? "代码生成失败");
      }
    } catch (e) {
      get().pushToast("error", e instanceof Error ? e.message : String(e));
    }
  },

  previewAssetCode: async (path) => {
    try {
      // Read an asset's doc, generate its Python, and push it to the code editor.
      const resp = await fetch(`/api/files/content?path=${encodeURIComponent(path)}`);
      const data = await resp.json();
      if (data.status !== "success") {
        get().pushToast("error", data.error ?? "读取失败");
        return;
      }
      const r2 = await fetch("/api/export_python", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data.doc),
      });
      const d2 = await r2.json();
      if (d2.status === "success") {
        set({ codeBuffer: d2.code as string });
        get().pushToast("success", "Code → Code Editor");
      } else {
        get().pushToast("error", d2.error ?? "代码生成失败");
      }
    } catch (e) {
      get().pushToast("error", e instanceof Error ? e.message : String(e));
    }
  },

  newWorkflow: (name) => {
    // Snapshot the current workflow's graph before switching to a fresh one.
    const cur = get().activeWorkflowId;
    const workflows = cur
      ? get().workflows.map((w) =>
          w.id === cur ? { ...w, nodes: get().nodes, edges: get().edges } : w
        )
      : get().workflows;
    const id = nextId("wf");
    set({
      workflows: [...workflows, { id, name, nodes: [], edges: [] }],
      activeWorkflowId: id,
      nodes: [],
      edges: [],
      selectedNodeId: null,
      results: {},
      status: "idle",
      dirty: false,
    });
    get().log(`新建工作流: ${name} (${id})`);
  },

  switchWorkflow: (id) => {
    // Persist the outgoing workflow's graph, then restore the target's.
    const cur = get().activeWorkflowId;
    let workflows = get().workflows;
    if (cur) {
      workflows = workflows.map((w) =>
        w.id === cur ? { ...w, nodes: get().nodes, edges: get().edges } : w
      );
    }
    const wf = workflows.find((w) => w.id === id);
    if (wf) {
      set({
        workflows,
        activeWorkflowId: id,
        nodes: wf.nodes,
        edges: wf.edges,
        selectedNodeId: null,
        results: {},
        status: "idle",
        dirty: false,
      });
      get().log(`切换到工作流: ${wf.name}`);
    }
  },

  refreshFiles: async () => {
    try {
      // Reload the server-side file list into state for the browser panels.
      const resp = await fetch("/api/files");
      const data = await resp.json();
      if (Array.isArray(data.files)) {
        set({ fileList: data.files });
      } else {
        get().log("文件列表格式错误");
      }
    } catch (e) {
      get().log(`文件列表加载失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  },

  openFile: async (path) => {
    try {
      // Load a `.riko`/`.ric` doc from the server and make it the active graph.
      const resp = await fetch(`/api/files/content?path=${encodeURIComponent(path)}`);
      const data = await resp.json();
      if (data.status === "success") {
        const { nodes, edges } = docToReactFlow(data.doc, get().nodeDefs);
        set({
          nodes,
          edges,
          selectedNodeId: null,
          results: {},
          status: "idle",
          activeFileName: data.doc?.metadata?.name ?? path,
          activeFilePath: path,
          graphStack: [{ name: data.doc?.metadata?.name ?? path.split("/").pop() ?? path, path, nodes, edges: [] }],
          dirty: false,
        });
        const recent = [path, ...get().recentFiles.filter((p) => p !== path)].slice(0, 10);
        saveRecent(recent);
        set({ recentFiles: recent });
        get().log(`已打开 ${path}`);
      } else {
        get().log(`打开失败: ${data.error}`);
      }
    } catch (e) {
      get().log(`打开失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  },

  openProjectFile: async (path) => {
    try {
      // Open a file via the project API (used for multi-file project documents).
      const resp = await fetch(`/api/project/open?path=${encodeURIComponent(path)}`);
      const data = await resp.json();
      if (data.status === "success") {
        const { nodes, edges } = docToReactFlow(data.doc, get().nodeDefs);
        const base = path.split("/").pop()?.replace(/\.(riko|ric)$/, "") ?? path;
        set({
          nodes,
          edges,
          selectedNodeId: null,
          results: {},
          status: "idle",
          activeFileName: base,
          activeFilePath: path,
          graphStack: [{ name: base, path, nodes, edges: [] }],
          dirty: false,
        });
        get().log(`已打开 ${path}`);
      } else {
        get().log(`打开失败: ${data.error}`);
      }
    } catch (e) {
      get().log(`打开失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  },

  importFileInline: async (path) => {
    try {
      const resp = await fetch(`/api/files/content?path=${encodeURIComponent(path)}`);
      const data = await resp.json();
      if (data.status !== "success") {
        get().log(`展开失败: ${data.error}`);
        return;
      }
      // Inline the referenced graph below the existing nodes, offset by maxY,
      // and remap its ids so they cannot collide with the current graph.
      const doc = data.doc as { nodes?: DocNode[]; edges?: DocEdge[] };
      const existing = get().nodes;
      let maxY = 0;
      for (const n of existing) maxY = Math.max(maxY, n.position.y + 120);
      const idMap: Record<string, string> = {};
      const inlined: Node<ThnNodeData>[] = (doc.nodes ?? []).map((n) => {
        const nid = nextId("n");
        idMap[n.id] = nid;
        const def = get().nodeDefs.find((d) => d.type === n.type_name);
        return {
          id: nid,
          type: "thnNode",
          position: { x: n.position?.[0] ?? 0, y: (n.position?.[1] ?? 0) + maxY },
          data: {
            type: n.type_name,
            label: n.label || def?.label || n.type_name,
            category: n.category || def?.category || "",
            params: n.parameters || {},
            inputs: def?.inputs ?? [],
            outputs: def?.outputs ?? [],
            hasError: false,
          },
        };
      });
      const inlinedEdges: Edge[] = (doc.edges ?? [])
        .map((e) => ({
          id: nextId("edge"),
          source: idMap[e.source_node],
          target: idMap[e.target_node],
          sourceHandle: e.source_port,
          targetHandle: e.target_port,
        }))
        .filter((e) => e.source && e.target);
      set({
        nodes: [...existing, ...inlined],
        edges: [...get().edges, ...inlinedEdges],
        dirty: true,
      });
      get().log(`已展开 ${path}（${inlined.length} 节点, ${inlinedEdges.length} 连接）`);
    } catch (e) {
      get().log(`展开失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  },

  saveFileToDisk: async (name, format = "ascii") => {
    // Save the graph to the server's managed file area (name + format).
    const doc = buildDoc(get().nodes, get().edges);
    const full = { ...doc, metadata: { name, description: "" } };
    try {
      const resp = await fetch("/api/files/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, format, doc: full }),
      });
      const data = await resp.json();
      if (data.status === "success") {
        set({ activeFileName: name, dirty: false });
        get().log(`已保存到 ${data.path} (${data.format === "binary" ? ".ric 二进制" : ".riko ASCII"})`);
        get().refreshFiles();
      } else {
        get().log(`保存失败: ${data.error ?? "未知错误"}`);
      }
    } catch (e) {
      get().log(`保存失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  },

  saveToPath: async (path, format = "ascii") => {
    // Save the graph to an explicit filesystem path (from the file picker).
    const doc = buildDoc(get().nodes, get().edges);
    try {
      const resp = await fetch("/api/fs/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, format, doc }),
      });
      const data = await resp.json();
      if (data.status === "success") {
        const name = data.name ?? path.split("/").pop()?.replace(/\.(riko|ric)$/i, "") ?? "workflow";
        set({ activeFileName: name, activeFilePath: path, dirty: false });
        get().log(`已保存到 ${data.path}`);
        get().pushToast("success", `Saved ${name}`);
        get().refreshFiles();
      } else {
        get().pushToast("error", data.error ?? "保存失败");
      }
    } catch (e) {
      get().pushToast("error", e instanceof Error ? e.message : String(e));
    }
  },

  importModule: (spec, position) => {
    // Place an `import` (module) or `graph_reference` (file) node referencing `spec`.
    const def =
      get().nodeDefs.find((n) => n.type === "import") ??
      get().nodeDefs.find((n) => n.type === "graph_reference");
    if (!def) {
      get().log("未找到 import / graph_reference 节点定义，请确认 API server 已启动");
      return;
    }
    const id = nextId(def.type);
    const params: Record<string, unknown> = {};
    for (const p of def.parameters) params[p.name] = p.default;
    params[def.type === "import" ? "module" : "file"] = spec;
    const node: Node<ThnNodeData> = {
      id,
      type: "thnNode",
      position,
      data: {
        type: def.type,
        label: def.label,
        category: def.category,
        params,
        inputs: def.inputs,
        outputs: def.outputs,
        hasError: false,
      },
    };
    set({ nodes: [...get().nodes, node], selectedNodeId: id, dirty: true });
    get().log(`导入 ${spec} (${def.type})`);
  },

  train: async (steps = 20, lr = 1e-3) => {
    set({ status: "running", losses: [] });
    get().log(`开始训练 (${steps} 步, lr=${lr})...`);
    try {
      // Stream newline-delimited JSON loss/error events from the training endpoint.
      const doc = buildDoc(get().nodes, get().edges);
      const resp = await fetch("/api/train/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc, steps, lr }),
      });
      if (!resp.body) throw new Error("响应无流 body");
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          const s = line.trim();
          if (!s) continue;
          let data: { loss?: number; error?: string; done?: boolean };
          try {
            data = JSON.parse(s);
          } catch {
            continue;
          }
          if (data.error) {
            get().log(`训练失败: ${data.error}`);
            get().pushToast("error", `训练失败: ${data.error}`);
            set({ status: "error" });
            return;
          }
          if (typeof data.loss === "number") {
            set({ losses: [...get().losses, data.loss] });
          }
          if (data.done) set({ status: "success" });
        }
      }
      const last = get().losses[get().losses.length - 1];
      get().log(`训练完成，最终 loss=${last?.toFixed(4) ?? "n/a"}`);
      set({ status: "success" });
    } catch (e) {
      set({ status: "error" });
      get().log(`训练失败: ${e instanceof Error ? e.message : String(e)}`);
      get().pushToast("error", `训练失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  },

  newBlank: () => {
    // Reset to an empty root-level graph.
    set({
      nodes: [],
      edges: [],
      frames: [],
      graphStack: [{ name: "root", path: "", nodes: [], edges: [] }],
      selectedNodeId: null,
      results: {},
      status: "idle",
      activeFileName: null,
      activeFilePath: null,
      dirty: false,
    });
    get().log("新建空白工作流");
  },

  setWelcomeOpen: (v) => set({ welcomeOpen: v }),

  setPrefsOpen: (v) => set({ prefsOpen: v }),

  setAboutOpen: (v) => set({ aboutOpen: v }),

  setImportFolderOpen: (v) => set({ importFolderOpen: v }),

  setFilePicker: (open, mode) => set({ filePickerOpen: open, filePickerMode: mode }),

  bumpProject: () => set({ projectVersion: get().projectVersion + 1 }),

  setCodeBuffer: (v) => set({ codeBuffer: v }),

  pushToast: (kind, message) => {
    // Keep at most 6 toasts; auto-dismiss non-error toasts after 6s.
    const id = ++toastUid;
    set({ toasts: [...get().toasts, { id, kind, message }].slice(-6) });
    if (kind !== "error") {
      setTimeout(() => get().dismissToast(id), 6000);
    }
  },

  dismissToast: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),

  log: (msg) => {
    // Prepend a timestamp and keep only the most recent 80 entries.
    const time = new Date().toLocaleTimeString();
    set({ logs: [...get().logs, `[${time}] ${msg}`].slice(-80) });
  },
}));
