/** Bottom-right toast stack for info / success / error notifications. */
import { useGraphStore } from "../store/graphStore";

export function ToastStack() {
  const toasts = useGraphStore((s) => s.toasts);
  const dismiss = useGraphStore((s) => s.dismissToast);

  return (
    <div className="toast-stack">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`}>
          <span className="toast-msg">{t.message}</span>
          <button className="toast-close" title="Dismiss" onClick={() => dismiss(t.id)}>
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
