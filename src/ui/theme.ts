/** Theme system: light / dark / follow-system, persisted to localStorage.

Sets `data-theme` on `<html>` so the CSS variable tokens in `styles.css`
switch. "system" leaves the attribute unset and lets `prefers-color-scheme`
drive the dark tokens. This module is intentionally framework-light so it can
be reused by any app shell that adopts the same CSS token set.
*/
import { create } from "zustand";

export type ThemeMode = "light" | "dark" | "system" | "glass";
const KEY = "entropia_riko_theme";
const BG_KEY = "entropia_riko_background";

export function loadTheme(): ThemeMode {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" || v === "glass" ? v : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(mode: ThemeMode): void {
  if (mode === "system") {
    delete document.documentElement.dataset.theme;
  } else {
    document.documentElement.dataset.theme = mode;
  }
}

export function applyBackground(url: string): void {
  const body = document.body;
  if (url) {
    body.style.backgroundImage = `url("${url}")`;
    body.style.backgroundSize = "cover";
    body.style.backgroundPosition = "center";
  } else {
    body.style.backgroundImage = "";
    body.style.backgroundSize = "";
    body.style.backgroundPosition = "";
  }
}

function loadBackground(): string {
  try {
    return localStorage.getItem(BG_KEY) ?? "";
  } catch {
    return "";
  }
}

export const useThemeStore = create<{
  theme: ThemeMode;
  setTheme: (m: ThemeMode) => void;
  backgroundImage: string;
  setBackgroundImage: (url: string) => void;
}>((set) => ({
  theme: loadTheme(),
  setTheme: (m) => {
    applyTheme(m);
    try {
      localStorage.setItem(KEY, m);
    } catch {
      /* ignore */
    }
    set({ theme: m });
  },
  backgroundImage: loadBackground(),
  setBackgroundImage: (url) => {
    applyBackground(url);
    try {
      if (url) localStorage.setItem(BG_KEY, url);
      else localStorage.removeItem(BG_KEY);
    } catch {
      /* ignore */
    }
    set({ backgroundImage: url });
  },
}));

// Initialize on import.
applyTheme(useThemeStore.getState().theme);
applyBackground(useThemeStore.getState().backgroundImage);
