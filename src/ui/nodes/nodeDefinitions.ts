/** Frontend node definitions (mirrors src/nodes for the UI).
 * Stage 1 uses these for the node library and graph canvas. Execution is a
 * local mock in graphStore; the Python runtime is connected in a later stage. */

export interface PortDef {
  name: string;
  label: string;
  dataKind: string;
  direction: "in" | "out";
}

export interface ParamDef {
  name: string;
  label: string;
  kind: "scalar" | "any";
  default: unknown;
  required: boolean;
  dtype?: string;
}

export interface NodeDef {
  type: string;
  label: string;
  category: string;
  inputs: PortDef[];
  outputs: PortDef[];
  parameters: ParamDef[];
}

export const NODE_DEFINITIONS: NodeDef[] = [
  {
    type: "constant",
    label: "Constant",
    category: "Inputs",
    inputs: [],
    outputs: [{ name: "value", label: "Value", dataKind: "tensor", direction: "out" }],
    parameters: [{ name: "value", label: "Value", kind: "any", default: 1, required: true, dtype: "float" }],
  },
  {
    type: "add",
    label: "Add",
    category: "Math",
    inputs: [
      { name: "left", label: "Left", dataKind: "tensor", direction: "in" },
      { name: "right", label: "Right", dataKind: "tensor", direction: "in" },
    ],
    outputs: [{ name: "result", label: "Result", dataKind: "tensor", direction: "out" }],
    parameters: [],
  },
  {
    type: "multiply",
    label: "Multiply",
    category: "Math",
    inputs: [
      { name: "left", label: "Left", dataKind: "tensor", direction: "in" },
      { name: "right", label: "Right", dataKind: "tensor", direction: "in" },
    ],
    outputs: [{ name: "result", label: "Result", dataKind: "tensor", direction: "out" }],
    parameters: [],
  },
];

export function getNodeDef(type: string): NodeDef | undefined {
  return NODE_DEFINITIONS.find((n) => n.type === type);
}

export const CATEGORIES = ["Inputs", "Math"];
