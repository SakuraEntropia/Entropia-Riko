"""Plugin loading for Entropia Riko.

A plugin is a directory under `plugins/` containing a `plugin.json` manifest
and (optionally) Python entry modules that register nodes. See
`docs/PLUGINS.md` for the interface and format.
"""

from .loader import PLUGINS_DIR, load_plugins, loaded_plugins  # noqa: F401
