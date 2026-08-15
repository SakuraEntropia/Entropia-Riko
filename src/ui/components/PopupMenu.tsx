/** Unified popup menu API.

One component powers every menu: the node right-click menu, the workspace "+"
menu, workspace-tab context menu, and (optionally) type dropdowns. Entries can
be grouped by `category`, carry a `description`, be `danger`/`disabled`, or act
as a `separator`. Rendered through a portal so it escapes frosted-glass
ancestors.
*/
import { useMemo, useState } from "react";
import { createPortal } from "react-dom";

export interface MenuEntry {
  id: string;
  label: string;
  description?: string;
  category?: string;
  danger?: boolean;
  disabled?: boolean;
  separator?: boolean;
  onSelect?: () => void;
}

export function PopupMenu({
  entries,
  x,
  y,
  onClose,
  searchable = false,
  maxWidth = 280,
}: {
  entries: MenuEntry[];
  x: number;
  y: number;
  onClose: () => void;
  searchable?: boolean;
  maxWidth?: number;
}) {
  const [query, setQuery] = useState("");
  // Tracks which categories the user has manually collapsed (by category name).
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const grouped = useMemo(() => {
    const q = query.trim().toLowerCase();
    // Filter by label/description only while searching; otherwise keep all entries.
    const filtered = q
      ? entries.filter(
          (e) =>
            !e.separator &&
            (e.label.toLowerCase().includes(q) ||
              (e.description ?? "").toLowerCase().includes(q))
        )
      : entries;
    const map: Record<string, MenuEntry[]> = {};
    const order: string[] = [];
    // Bucket entries by category ("" for uncategorized) preserving first-seen order.
    for (const e of filtered) {
      const cat = e.category ?? "";
      if (!(cat in map)) {
        map[cat] = [];
        order.push(cat);
      }
      map[cat].push(e);
    }
    return { map, order };
  }, [entries, query]);

  const { map, order } = grouped;

  return createPortal(
    <>
      <div
        className="popup-overlay"
        onClick={onClose}
        onContextMenu={(e) => {
          e.preventDefault();
          onClose();
        }}
      />
      <div className="popup-menu" style={{ left: x, top: y, maxWidth }}>
        {searchable && (
          <div className="popup-search">
            <input
              autoFocus
              placeholder="Search…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
        )}
        <div className="popup-list">
          {order.map((cat) => {
            // Ignore manual collapse state while searching so matches stay visible.
            const isCollapsed = !query && !!collapsed[cat];
            return (
              <div key={cat} className="popup-category">
                {cat !== "" && (
                  <div
                    className="popup-cat-header"
                    onClick={() => setCollapsed((c) => ({ ...c, [cat]: !c[cat] }))}
                  >
                    <span className="popup-cat-toggle">{isCollapsed ? "▸" : "▾"}</span>
                    <span className="popup-cat-label">{cat}</span>
                    <span className="popup-cat-count">{map[cat].length}</span>
                  </div>
                )}
                {!isCollapsed &&
                  map[cat].map((e) =>
                    e.separator ? (
                      <div key={e.id} className="popup-sep" />
                    ) : (
                      <div
                        key={e.id}
                        className={`popup-item ${e.danger ? "danger" : ""} ${e.disabled ? "disabled" : ""}`}
                        onClick={() => {
                          if (e.disabled) return;
                          e.onSelect?.();
                          onClose();
                        }}
                      >
                        <span className="popup-item-label">{e.label}</span>
                        {e.description && <span className="popup-item-desc">{e.description}</span>}
                      </div>
                    )
                  )}
              </div>
            );
          })}
          {order.length === 0 && <div className="popup-item disabled">No matches</div>}
        </div>
      </div>
    </>,
    document.body
  );
}
