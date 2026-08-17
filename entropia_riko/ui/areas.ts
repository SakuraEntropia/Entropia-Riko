/** Blender-style area tree.

The window layout is a binary tree of splits. Each leaf is an "area" (a panel
showing one editor type). Splits are either a row (children side-by-side,
separated by a vertical divider) or a column (children stacked, separated by a
horizontal divider). This lets every panel be split both horizontally and
vertically, and merged away, exactly like Blender's editor areas.
*/
import type { PanelType } from "./components/Panel";

export type AreaNode =
  | { kind: "leaf"; id: string; type: PanelType }
  | {
      kind: "split";
      id: string;
      direction: "row" | "column";
      first: AreaNode;
      second: AreaNode;
      ratio: number;
    };

let _uid = 0;
const uid = () => `area_${++_uid}`;

export const leaf = (type: PanelType): AreaNode => ({
  kind: "leaf",
  id: uid(),
  type,
});

export const split = (
  direction: "row" | "column",
  first: AreaNode,
  second: AreaNode,
  ratio = 0.5
): AreaNode => ({ kind: "split", id: uid(), direction, first, second, ratio });

/** Default layout: [nodes/files stacked] | [canvas | (New File over inspector)] over [status | loss]. */
export function defaultLayout(): AreaNode {
  const left = split("column", leaf("nodes"), leaf("files"), 0.72);
  // "New File" sits directly above the inspector (rightmost column only), never
  // over the canvas/graph area.
  const right = split(
    "row",
    leaf("canvas"),
    split("column", leaf("project"), leaf("inspector"), 0.3),
    0.68
  );
  const main = split("row", left, right, 0.26);
  const bottom = split("row", leaf("status"), leaf("loss"), 0.55);
  return split("column", main, bottom, 0.82);
}

/** Shared skeleton for workflow-specific layouts (left column + canvas + right). */
function buildLayout(cfg: {
  right: PanelType;
  bottom: PanelType[];
  leftExtra?: PanelType;
  rightTop?: PanelType;
  canvasRatio?: number;
  leftRatio?: number;
  colRatio?: number;
  rightTopRatio?: number;
}): AreaNode {
  const left = cfg.leftExtra
    ? split("column", leaf("nodes"), leaf(cfg.leftExtra), 0.7)
    : leaf("nodes");
  let rightInner: AreaNode = leaf(cfg.right);
  if (cfg.rightTop) {
    rightInner = split("column", leaf(cfg.rightTop), rightInner, cfg.rightTopRatio ?? 0.28);
  }
  const right = split("row", leaf("canvas"), rightInner, cfg.canvasRatio ?? 0.68);
  const main = split("row", left, right, cfg.leftRatio ?? 0.26);
  const bottom = cfg.bottom.length === 1
    ? leaf(cfg.bottom[0])
    : split("row", leaf(cfg.bottom[0]), leaf(cfg.bottom[1]), 0.55);
  return split("column", main, bottom, cfg.colRatio ?? 0.84);
}

/** A named workspace preset (categorized for the "+" menu). */
export interface WorkspacePreset {
  id: string;
  label: string;
  category: string;
  description: string;
  build: () => AreaNode;
}

export const WORKSPACE_PRESETS: WorkspacePreset[] = [
  { id: "layout", label: "Layout", category: "General", description: "Full IDE: nodes, asset library, code editor, canvas, inspector.",
    build: defaultLayout },
  { id: "code", label: "Code", category: "General", description: "Big code editor + canvas for previewing exported PyTorch.",
    build: () => buildLayout({ right: "code", bottom: ["status", "loss"], leftExtra: "files", canvasRatio: 0.6 }) },
  { id: "inference", label: "Inference", category: "General", description: "Large canvas + inspector, no training.",
    build: () => buildLayout({ right: "inspector", bottom: ["status", "docs"], leftExtra: "files", canvasRatio: 0.74 }) },
  { id: "training", label: "Training", category: "Training", description: "Live loss curve + inspector for training runs.",
    build: () => buildLayout({ right: "loss", bottom: ["status", "inspector"], leftExtra: "files", canvasRatio: 0.6 }) },
  { id: "tuning", label: "Hyperparameter Tuning", category: "Training", description: "Loss + docs; inspector below.",
    build: () => buildLayout({ right: "loss", bottom: ["inspector", "docs"], canvasRatio: 0.62 }) },
  { id: "mnist_studio", label: "MNIST Studio", category: "Vision", description: "Handwriting pad above the inspector — draw a digit and infer.",
    build: () => buildLayout({ right: "inspector", rightTop: "pad", bottom: ["status", "loss"], leftExtra: "files", canvasRatio: 0.58 }) },
  { id: "image_classifier", label: "Image Classifier", category: "Vision", description: "Image previews + loss.",
    build: () => buildLayout({ right: "inspector", bottom: ["status", "loss"], leftExtra: "files", canvasRatio: 0.68 }) },
  { id: "image_gen", label: "Text → Image", category: "Vision", description: "Project explorer above inspector for Diffusers generation.",
    build: () => buildLayout({ right: "inspector", rightTop: "project", bottom: ["status"], canvasRatio: 0.62 }) },
  { id: "detection", label: "Object Detection", category: "Vision", description: "Canvas + inspector + docs.",
    build: () => buildLayout({ right: "inspector", bottom: ["status", "docs"], leftExtra: "files", canvasRatio: 0.7 }) },
  { id: "text_classifier", label: "Text Classifier", category: "Language", description: "Text previews + docs.",
    build: () => buildLayout({ right: "inspector", bottom: ["status", "docs"], leftExtra: "files", canvasRatio: 0.7 }) },
  { id: "text_gen", label: "Text Gen", category: "Language", description: "Text preview for Transformers.",
    build: () => buildLayout({ right: "inspector", bottom: ["status"], rightTop: "project", canvasRatio: 0.68 }) },
  { id: "embeddings", label: "Embeddings / JEPA", category: "Language", description: "Pretrained encoder embeddings.",
    build: () => buildLayout({ right: "inspector", bottom: ["status", "docs"], canvasRatio: 0.72 }) },
];

/** A runtime workspace instance (its own editable layout). */
export interface WorkspaceInstance {
  id: string;
  name: string;
  root: AreaNode;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Replace a leaf with a split that keeps the leaf + a new sibling of the same type. */
export function splitLeaf(
  node: AreaNode,
  id: string,
  direction: "row" | "column"
): AreaNode {
  if (node.kind === "leaf") {
    return node.id === id ? split(direction, node, leaf(node.type), 0.5) : node;
  }
  return {
    ...node,
    first: splitLeaf(node.first, id, direction),
    second: splitLeaf(node.second, id, direction),
  };
}

/** Remove the divider around a node (replace its parent split with its sibling). */
export function closeLeaf(node: AreaNode, id: string): AreaNode {
  // Remove the leaf `id`; its sibling absorbs the space (used by the ✕ button).
  if (node.kind === "split") {
    if (node.first.id === id) return node.second;
    if (node.second.id === id) return node.first;
    return {
      ...node,
      first: closeLeaf(node.first, id),
      second: closeLeaf(node.second, id),
    };
  }
  return node;
}

/** Join: keep the dragged leaf `id`, remove its sibling (Blender drag-outward). */
export function mergeLeaf(node: AreaNode, id: string): AreaNode {
  if (node.kind === "split") {
    if (node.first.id === id) return node.first;
    if (node.second.id === id) return node.second;
    return {
      ...node,
      first: mergeLeaf(node.first, id),
      second: mergeLeaf(node.second, id),
    };
  }
  return node;
}

/** Return the id of `id`'s sibling (the other child of its parent split), if any. */
export function siblingNodeId(node: AreaNode, id: string): string | null {
  if (node.kind === "split") {
    if (node.first.id === id) return node.second.id;
    if (node.second.id === id) return node.first.id;
    const a = siblingNodeId(node.first, id);
    if (a !== null) return a;
    return siblingNodeId(node.second, id);
  }
  return null;
}

/** Adjust a split's ratio (delta is a fraction of the split's own extent). */
export function resizeSplit(node: AreaNode, id: string, delta: number): AreaNode {
  if (node.kind === "split") {
    if (node.id === id) {
      return { ...node, ratio: clamp(node.ratio + delta, 0.08, 0.92) };
    }
    return {
      ...node,
      first: resizeSplit(node.first, id, delta),
      second: resizeSplit(node.second, id, delta),
    };
  }
  return node;
}

/** Change the editor type of a leaf. */
export function setLeafType(
  node: AreaNode,
  id: string,
  type: PanelType
): AreaNode {
  if (node.kind === "leaf") {
    return node.id === id ? { ...node, type } : node;
  }
  return {
    ...node,
    first: setLeafType(node.first, id, type),
    second: setLeafType(node.second, id, type),
  };
}

export function countLeaves(node: AreaNode): number {
  if (node.kind === "leaf") return 1;
  return countLeaves(node.first) + countLeaves(node.second);
}
