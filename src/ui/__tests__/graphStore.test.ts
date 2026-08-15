/** Frontend unit tests for graphStore (vitest). */
import { describe, it, expect, beforeEach } from "vitest";
import { useGraphStore, apiNodeToDef } from "../store/graphStore";

describe("apiNodeToDef", () => {
  it("converts an API node definition to frontend NodeDef format", () => {
    const def = apiNodeToDef({
      type: "add",
      label: "Add",
      category: "Math",
      inputs: [{ name: "left", data_kind: "tensor", required: true }],
      outputs: [{ name: "result", data_kind: "tensor" }],
      parameters: [
        { name: "value", kind: "scalar", default: 0, required: false, dtype: "int" },
      ],
    });
    expect(def.type).toBe("add");
    expect(def.label).toBe("Add");
    expect(def.category).toBe("Math");
    expect(def.inputs[0].name).toBe("left");
    expect(def.inputs[0].label).toBe("Left");
    expect(def.inputs[0].direction).toBe("in");
    expect(def.outputs[0].label).toBe("Result");
    expect(def.outputs[0].direction).toBe("out");
    expect(def.parameters[0].dtype).toBe("int");
  });

  it("handles null dtype -> undefined", () => {
    const def = apiNodeToDef({
      type: "x",
      label: "X",
      category: "T",
      inputs: [],
      outputs: [],
      parameters: [{ name: "p", kind: "scalar", default: null, required: false, dtype: null }],
    });
    expect(def.parameters[0].dtype).toBeUndefined();
  });
});

describe("graphStore", () => {
  beforeEach(() => {
    useGraphStore.setState({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      results: {},
      status: "idle",
      logs: [],
    });
  });

  it("addNode creates a node in the store", () => {
    useGraphStore.getState().addNode("constant", { x: 10, y: 20 });
    const { nodes } = useGraphStore.getState();
    expect(nodes).toHaveLength(1);
    expect(nodes[0].data.type).toBe("constant");
    expect(nodes[0].data.label).toBe("Constant");
    expect(nodes[0].position).toEqual({ x: 10, y: 20 });
  });

  it("addNode selects the new node", () => {
    useGraphStore.getState().addNode("add", { x: 0, y: 0 });
    expect(useGraphStore.getState().selectedNodeId).not.toBeNull();
  });

  it("onConnect adds an edge between two nodes", () => {
    const s = useGraphStore.getState();
    s.addNode("constant", { x: 0, y: 0 });
    s.addNode("add", { x: 100, y: 0 });
    const ids = useGraphStore.getState().nodes.map((n) => n.id);
    s.onConnect({
      source: ids[0],
      target: ids[1],
      sourceHandle: "value",
      targetHandle: "left",
    });
    expect(useGraphStore.getState().edges).toHaveLength(1);
  });

  it("removeNode removes node and cascades edges", () => {
    const s = useGraphStore.getState();
    s.addNode("constant", { x: 0, y: 0 });
    s.addNode("add", { x: 100, y: 0 });
    const ids = useGraphStore.getState().nodes.map((n) => n.id);
    s.onConnect({ source: ids[0], target: ids[1], sourceHandle: "value", targetHandle: "left" });
    s.removeNode(ids[0]);
    const st = useGraphStore.getState();
    expect(st.nodes).toHaveLength(1);
    expect(st.edges).toHaveLength(0);
  });

  it("load restores nodes and edges from a workflow JSON", async () => {
    const json = JSON.stringify({
      version: "0.1",
      nodes: [
        {
          id: "n1",
          type_name: "constant",
          label: "Constant",
          category: "Inputs",
          position: [10, 20],
          parameters: { value: 5 },
          inputs: [],
          outputs: [],
        },
      ],
      edges: [],
      settings: {},
    });
    await useGraphStore.getState().load(new File([json], "w.json"));
    const { nodes } = useGraphStore.getState();
    expect(nodes).toHaveLength(1);
    expect(nodes[0].id).toBe("n1");
    expect(nodes[0].data.type).toBe("constant");
    expect(nodes[0].position).toEqual({ x: 10, y: 20 });
  });

  it("updateParam updates a node parameter", () => {
    useGraphStore.getState().addNode("constant", { x: 0, y: 0 });
    const id = useGraphStore.getState().nodes[0].id;
    useGraphStore.getState().updateParam(id, "value", 42);
    expect(useGraphStore.getState().nodes[0].data.params.value).toBe(42);
  });
});
