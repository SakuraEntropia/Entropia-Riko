/** Built-in documentation browser (Blender/Houdini-style web manual).

Supports language switching (English / 中文). The choice is persisted to
localStorage and defaults to English.
*/
import { useEffect, useMemo, useState } from "react";
import guideZh from "../../../docs/USER_GUIDE.md?raw";
import guideEn from "../../../docs/USER_GUIDE_EN.md?raw";
import { renderMarkdown } from "../docs/Markdown";

interface Section {
  id: string;
  title: string;
  body: string;
}

const LANG_KEY = "entropia_riko_doc_lang";
type Lang = "en" | "zh";

function parseGuide(raw: string): { title: string; intro: string; sections: Section[] } {
  const lines = raw.split("\n");
  let title = "";
  const sections: Section[] = [];
  let current: Section | null = null;
  const intro: string[] = [];

  for (const line of lines) {
    if (line.startsWith("# ")) {
      title = line.slice(2).trim();
      continue;
    }
    if (line.startsWith("## ")) {
      if (current) sections.push(current);
      current = { id: line.slice(3).trim(), title: line.slice(3).trim(), body: "" };
      continue;
    }
    if (current) {
      current.body += line + "\n";
    } else if (line.trim() !== "") {
      intro.push(line);
    }
  }
  if (current) sections.push(current);
  return { title, intro: intro.join("\n"), sections };
}

function loadLang(): Lang {
  try {
    return localStorage.getItem(LANG_KEY) === "zh" ? "zh" : "en";
  } catch {
    return "en";
  }
}

export function DocsPanel() {
  const [lang, setLang] = useState<Lang>(loadLang);
  const doc = useMemo(
    () => parseGuide(lang === "zh" ? guideZh : guideEn),
    [lang]
  );
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string>(doc.sections[0]?.id ?? "");

  useEffect(() => {
    try {
      localStorage.setItem(LANG_KEY, lang);
    } catch {
      /* ignore */
    }
  }, [lang]);

  // keep selection valid when switching language (ids match between languages)
  useEffect(() => {
    setSelected((prev) =>
      doc.sections.some((s) => s.id === prev) ? prev : (doc.sections[0]?.id ?? "")
    );
  }, [doc]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return doc.sections;
    return doc.sections.filter(
      (s) => s.title.toLowerCase().includes(q) || s.body.toLowerCase().includes(q)
    );
  }, [doc.sections, query]);

  const idx = doc.sections.findIndex((s) => s.id === selected);
  const current = doc.sections[idx] ?? doc.sections[0];
  const prev = idx > 0 ? doc.sections[idx - 1] : null;
  const next = idx >= 0 && idx < doc.sections.length - 1 ? doc.sections[idx + 1] : null;

  return (
    <div className="docs-panel">
      <div className="docs-sidebar">
        <div className="docs-search">
          <input
            placeholder={lang === "zh" ? "搜索文档…" : "Search docs…"}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="docs-nav">
          {filtered.map((s) => (
            <div
              key={s.id}
              className={`docs-nav-item ${s.id === selected ? "active" : ""}`}
              onClick={() => setSelected(s.id)}
            >
              {s.title}
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="docs-nav-empty">
              {lang === "zh" ? "无匹配" : "No matches"}
            </div>
          )}
        </div>
      </div>
      <div className="docs-content">
        <div className="docs-content-inner">
          <div className="docs-langbar">
            <div className="docs-pager-spacer" />
            <div className="docs-lang-switch">
              <button
                className={`panel-btn ${lang === "en" ? "active" : ""}`}
                onClick={() => setLang("en")}
              >
                EN
              </button>
              <button
                className={`panel-btn ${lang === "zh" ? "active" : ""}`}
                onClick={() => setLang("zh")}
              >
                中文
              </button>
            </div>
          </div>
          {query ? (
            <h1>{lang === "zh" ? "搜索结果" : "Search Results"}</h1>
          ) : (
            <>
              <h1>{doc.title}</h1>
              <div className="docs-intro">{renderMarkdown(doc.intro)}</div>
              <hr />
            </>
          )}
          {query ? (
            filtered.map((s) => (
              <div key={s.id} className="docs-result">
                <h2 onClick={() => { setQuery(""); setSelected(s.id); }}>{s.title}</h2>
                {renderMarkdown(s.body.slice(0, 400))}
              </div>
            ))
          ) : (
            current && (
              <>
                <h1>{current.title}</h1>
                {renderMarkdown(current.body)}
              </>
            )
          )}
          {!query && (
            <div className="docs-pager">
              {prev && (
                <button className="btn btn-sm" onClick={() => setSelected(prev.id)}>
                  ← {prev.title}
                </button>
              )}
              <div className="docs-pager-spacer" />
              {next && (
                <button className="btn btn-sm" onClick={() => setSelected(next.id)}>
                  {next.title} →
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
