/** Minimal Markdown → JSX renderer (no external dependency).

Handles the constructs used in the built-in documentation: headings (h1-h4),
code fences, inline code, bold, links, blockquotes, ordered/unordered lists,
tables, and horizontal rules.
*/
import { Fragment } from "react";

function renderInline(text: string): React.ReactNode[] {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g);
  return parts.map((part, i) => {
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={i}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    const m = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (m) {
      return (
        <a key={i} href={m[2]} target="_blank" rel="noreferrer">
          {m[1]}
        </a>
      );
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

function isHeading(line: string): boolean {
  return /^#{1,4}\s/.test(line);
}

function isListItem(line: string): boolean {
  return /^\s*[-*]\s/.test(line) || /^\s*\d+\.\s/.test(line);
}

function isTableSeparator(line: string): boolean {
  return /^\s*\|[\s:|-]+\|\s*$/.test(line);
}

export function renderMarkdown(md: string): React.ReactNode[] {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;

  const push = (el: React.ReactNode, key: number) => {
    blocks.push(<Fragment key={key}>{el}</Fragment>);
  };

  while (i < lines.length) {
    const line = lines[i];

    // blank line
    if (line.trim() === "") {
      i++;
      continue;
    }

    // code fence
    if (line.trim().startsWith("```")) {
      const code: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]);
        i++;
      }
      i++; // skip closing fence
      push(<pre><code>{code.join("\n")}</code></pre>, i);
      continue;
    }

    // heading
    if (isHeading(line)) {
      const level = line.match(/^(#+)/)![1].length;
      const text = line.replace(/^#+\s*/, "").trim();
      const key = i;
      if (level === 1) push(<h1>{renderInline(text)}</h1>, key);
      else if (level === 2) push(<h2>{renderInline(text)}</h2>, key);
      else if (level === 3) push(<h3>{renderInline(text)}</h3>, key);
      else push(<h4>{renderInline(text)}</h4>, key);
      i++;
      continue;
    }

    // horizontal rule
    if (/^\s*-{3,}\s*$/.test(line)) {
      push(<hr />, i);
      i++;
      continue;
    }

    // blockquote
    if (line.startsWith(">")) {
      const q: string[] = [];
      while (i < lines.length && lines[i].startsWith(">")) {
        q.push(lines[i].replace(/^>\s?/, ""));
        i++;
      }
      push(<blockquote>{renderMarkdown(q.join("\n"))}</blockquote>, i);
      continue;
    }

    // table
    if (line.trim().startsWith("|") && lines[i + 1] && isTableSeparator(lines[i + 1])) {
      const header = line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
        i++;
      }
      push(
        <table>
          <thead>
            <tr>{header.map((h, idx) => <th key={idx}>{renderInline(h)}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri}>{r.map((c, ci) => <td key={ci}>{renderInline(c)}</td>)}</tr>
            ))}
          </tbody>
        </table>,
        i
      );
      continue;
    }

    // list
    if (isListItem(line)) {
      const ordered = /^\s*\d+\.\s/.test(line);
      const items: string[] = [];
      while (i < lines.length && isListItem(lines[i])) {
        items.push(lines[i].replace(/^\s*(?:[-*]|\d+\.)\s*/, ""));
        i++;
      }
      const key = i;
      push(
        ordered ? (
          <ol>{items.map((it, idx) => <li key={idx}>{renderInline(it)}</li>)}</ol>
        ) : (
          <ul>{items.map((it, idx) => <li key={idx}>{renderInline(it)}</li>)}</ul>
        ),
        key
      );
      continue;
    }

    // paragraph (accumulate until a block boundary)
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !isHeading(lines[i]) &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith("|") &&
      !lines[i].startsWith(">") &&
      !isListItem(lines[i]) &&
      !/^\s*-{3,}\s*$/.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    if (para.length) {
      push(<p>{renderInline(para.join(" "))}</p>, i);
    }
  }

  return blocks;
}
