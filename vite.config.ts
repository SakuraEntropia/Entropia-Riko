import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Entropia Riko — Vite config.
// root is the project dir; the app entry is /src/ui/main.tsx (index.html).
// Python sources under src/core, src/runtime, etc. are ignored by Vite.
export default defineConfig({
  plugins: [react()],
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
