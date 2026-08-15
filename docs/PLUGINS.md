# Plugins — Interface & Format

Plugins extend Entropia Riko without touching the core. A plugin is a directory
under `plugins/` containing a `plugin.json` manifest and (optionally) entry
modules.

## Directory layout

```
plugins/
  example_plugin/
    plugin.json      # manifest (required)
    nodes.py         # Python entry (registers node types)
    # ...any other files the plugin needs
```

The server scans `plugins/*/plugin.json` at startup and imports each plugin's
`entry` module.

## Manifest (`plugin.json`)

```json
{
  "name": "example_plugin",
  "version": "1.0.0",
  "description": "Example plugin: registers a `plugin_double` node.",
  "author": "…",
  "entry": "nodes.py",
  "requires": []
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✓ | Unique plugin id (directory name is used as a fallback). |
| `version` | string | ✓ | Semantic version. |
| `description` | string | | Short description. |
| `author` | string | | Author. |
| `entry` | string | | Python module to import (relative to the plugin directory). |
| `requires` | array | | Reserved for dependency declarations. |

## Backend interface (Python)

A plugin's `entry` module registers node types with the standard decorator. It
imports the public API via absolute `src.*` paths (the server runs from the
project root, so `src` is importable):

```python
# plugins/example_plugin/nodes.py
from src.runtime.registry import register
from src.nodes.base import BaseNode, NodeInput, NodeOutput, Parameter
from src.core.tensor import TensorValue


@register("plugin_double")
class PluginDoubleNode(BaseNode):
    type_name = "plugin_double"
    label = "Double (plugin)"
    category = "Plugin"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = [Parameter("scale", default=2.0, dtype="float")]

    def execute(self, inputs, params, context):
        ...
        return {"result": TensorValue(...)}
```

- The node appears in `/api/nodes` and the UI library under its `category`.
- Plugins may also register new panel window types via the frontend hook below.

A broken plugin (import error / missing manifest) is reported with
`status: "error"` and an `error` field, and never crashes the app — see
`GET /api/plugins`.

## Frontend interface (TypeScript)

The panel/area-tree system is pluggable. A frontend module can register a
renderer for a window type:

```ts
import { registerPanelContent } from "./components/Panel";

registerPanelContent("my_view", () => <MyViewComponent />);
```

The layout system (split / merge / resize / type switching) is reused unchanged.

> Note: frontend code is bundled at build time — a plugin that ships frontend
> code must be imported into the app (or loaded via dev-mode dynamic import).
> Backend (Python) plugins are loaded from disk at runtime.

## Discovery

- `GET /api/plugins` lists loaded plugins and their status.
- `GET /api/nodes` includes plugin-registered node types.

## Example

See `plugins/example_plugin/` for a minimal working plugin (registers
`plugin_double`: `x → 2x`).
