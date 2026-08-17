import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// Entropia Riko — Vite config.
// The UI is decoupled: the app entry is /frontend/main.tsx, which mounts the
// `entropia-template-ui` npm package (GitHub: SakuraEntropia/Entropia-Template-UI).
// Python sources under entropia_riko/ are ignored by Vite.

// App version comes from git so it stays in sync with the repo tags:
// `git describe --tags` → "v0.1.0" (on a tag) or "v0.1.0-3-g<sha>" (ahead).
// Without a .git dir (e.g. the release zip), fall back to package.json version.
function appVersion(): string {
  try {
    const v = execSync("git describe --tags --always --dirty", { encoding: "utf8" }).trim();
    return v.startsWith("v") ? v.slice(1) : v;
  } catch {
    try {
      const pkg = JSON.parse(readFileSync(join(process.cwd(), "package.json"), "utf8"));
      return pkg.version || "0.1.0";
    } catch {
      return "0.1.0";
    }
  }
}

export default defineConfig({
  plugins: [react()],
  define: {
    __APP_VERSION__: JSON.stringify(appVersion()),
  },
  root: ".",
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
