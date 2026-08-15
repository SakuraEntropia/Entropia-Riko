/** About dialog (floating window, opened from the app logo menu). */
import { APP_VERSION } from "../version";
import { FloatingWindow } from "./FloatingWindow";

export function AboutPanel({ onClose }: { onClose: () => void }) {
  return (
    <FloatingWindow title="About Entropia Riko" onClose={onClose} width={440}>
      <div className="about-hero">
        <img className="about-logo" src="/brand/logo.svg" alt="Entropia Riko logo" />
        <h2>Entropia Riko</h2>
        <div className="about-version">Version {APP_VERSION}</div>
      </div>
      <div className="about-body">
        <p className="prefs-desc">
          A professional node-graph deep-learning editor. Build models visually,
          run inference, train with a live loss curve, and export to clean
          PyTorch or TensorFlow/Keras code.
        </p>
        <table className="about-meta">
          <tbody>
            <tr><td>Backends</td><td>PyTorch · TensorFlow / Keras</td></tr>
            <tr><td>File formats</td><td>.riko (JSON) · .ric (binary zlib)</td></tr>
            <tr><td>Plugins</td><td>plugins/*/plugin.json (Python node modules)</td></tr>
            <tr><td>License</td><td>MIT</td></tr>
          </tbody>
        </table>
        <div className="about-assets">
          <span>Brand assets (replace in place):</span>
          <code>public/brand/logo.svg</code>
          <code>public/brand/hero.jpg</code>
        </div>
      </div>
    </FloatingWindow>
  );
}
