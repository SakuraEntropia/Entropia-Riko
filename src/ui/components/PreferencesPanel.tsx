/** Preferences dialog — macOS/Blender-style: category sidebar on the left,
 * settings for the selected category on the right. */
import { useEffect, useState } from "react";
import { useThemeStore, type ThemeMode } from "../theme";
import { APP_VERSION } from "../version";
import { PluginPanel } from "./PluginPanel";
import { FloatingWindow } from "./FloatingWindow";
import { BrandLogo } from "./BrandLogo";

type CategoryId = "appearance" | "plugins" | "about";

const CATEGORIES: { id: CategoryId; label: string; icon: string }[] = [
  { id: "appearance", label: "Appearance", icon: "◧" },
  { id: "plugins", label: "Plugins", icon: "⬡" },
  { id: "about", label: "About", icon: "ⓘ" },
];

export function PreferencesPanel({ onClose }: { onClose: () => void }) {
  const theme = useThemeStore((s) => s.theme);
  const setTheme = useThemeStore((s) => s.setTheme);
  const bg = useThemeStore((s) => s.backgroundImage);
  const setBg = useThemeStore((s) => s.setBackgroundImage);

  const [category, setCategory] = useState<CategoryId>("appearance");
  const [bgInput, setBgInput] = useState(bg);

  useEffect(() => {
    setBgInput(bg);
  }, [bg]);

  const themes: { value: ThemeMode; label: string; hint: string }[] = [
    { value: "light", label: "Light", hint: "Always use the light palette" },
    { value: "dark", label: "Dark", hint: "Always use the dark palette" },
    { value: "system", label: "System", hint: "Follow the OS color scheme" },
    { value: "glass", label: "Liquid Glass", hint: "Apple-style translucent glass" },
  ];

  return (
    <FloatingWindow title="Preferences" onClose={onClose} width={720} zIndex={1001}>
      <div className="prefs-layout">
        <nav className="prefs-sidebar">
          {CATEGORIES.map((c) => (
            <button
              key={c.id}
              className={`prefs-nav ${category === c.id ? "active" : ""}`}
              onClick={() => setCategory(c.id)}
            >
              <span className="prefs-nav-icon">{c.icon}</span>
              {c.label}
            </button>
          ))}
        </nav>

        <div className="prefs-content">
            {category === "appearance" && (
              <>
                <section className="prefs-section">
                  <h3>Theme</h3>
                  <p className="prefs-desc">Choose how the interface looks.</p>
                  <div className="prefs-theme-list">
                    {themes.map((t) => (
                      <label
                        key={t.value}
                        className={`prefs-theme ${theme === t.value ? "active" : ""}`}
                      >
                        <input
                          type="radio"
                          name="theme"
                          checked={theme === t.value}
                          onChange={() => setTheme(t.value)}
                        />
                        <span className="prefs-theme-swatch" data-theme={t.value} />
                        <span className="prefs-theme-label">{t.label}</span>
                        <span className="prefs-theme-hint">{t.hint}</span>
                      </label>
                    ))}
                  </div>
                </section>

                <section className="prefs-section">
                  <h3>Background image</h3>
                  <p className="prefs-desc">
                    A cover image behind the editor. URL is saved locally.
                  </p>
                  <div className="prefs-row">
                    <input
                      className="prefs-input"
                      placeholder="Image URL (https://…)"
                      value={bgInput}
                      onChange={(e) => setBgInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") setBg(bgInput.trim());
                      }}
                    />
                    <button className="btn btn-sm" onClick={() => setBg(bgInput.trim())}>
                      Apply
                    </button>
                    <button
                      className="btn btn-sm"
                      onClick={() => {
                        setBgInput("");
                        setBg("");
                      }}
                    >
                      Clear
                    </button>
                  </div>
                </section>
              </>
            )}

            {category === "plugins" && (
              <section className="prefs-section" style={{ marginBottom: 0 }}>
                <PluginPanel />
              </section>
            )}

            {category === "about" && (
              <section className="prefs-section">
                <div className="prefs-about">
                  <BrandLogo className="prefs-about-logo" />
                  <h3>Entropia Riko</h3>
                  <div className="prefs-about-version">Version {APP_VERSION}</div>
                  <p className="prefs-desc">
                    A professional node-graph deep-learning editor. Build models
                    visually, run inference, train with live loss curves, and
                    export to clean PyTorch or TensorFlow/Keras code.
                  </p>
                  <div className="prefs-about-meta">
                    <span>Backends: PyTorch · TensorFlow/Keras</span>
                    <span>Brand assets: public/brand/</span>
                  </div>
                </div>
              </section>
            )}
          </div>
        </div>
    </FloatingWindow>
  );
}
