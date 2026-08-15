/** Code preview modal: shows the exported PyTorch code for a .riko file. */
import { useEffect, useState } from "react";
import { useGraphStore } from "../store/graphStore";

export function CodePreview({
  path,
  name,
  onClose,
}: {
  path: string;
  name: string;
  onClose: () => void;
}) {
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const pushToast = useGraphStore((s) => s.pushToast);

  useEffect(() => {
    fetch("/api/project/code", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    })
      .then((r) => r.json())
      .then((d) => {
        if (d.status === "success") setCode(d.code ?? "");
        else setError(d.error ?? "failed to generate code");
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [path]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      pushToast("success", "Copied PyTorch code");
    } catch {
      pushToast("error", "Copy failed");
    }
  };

  return (
    <div className="prefs-overlay" onClick={onClose}>
      <div className="code-preview" onClick={(e) => e.stopPropagation()}>
        <div className="code-preview-head">
          <span className="code-preview-title">PyTorch · {name}</span>
          <span className="spacer" />
          <button className="panel-btn" onClick={copy}>Copy</button>
          <button className="panel-btn" onClick={onClose}>✕</button>
        </div>
        <pre className="code-preview-body">
          {error ? `// ${error}` : code || "// generating…"}
        </pre>
      </div>
    </div>
  );
}
