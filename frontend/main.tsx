/** Entropia Riko frontend entry — the UI is fully decoupled and ships as the
 * `entropia-template-ui` npm package (GitHub: SakuraEntropia/Entropia-Template-UI).
 * This file only mounts that editor and points it at this repo's API server.
 */
import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "entropia-template-ui";
import "entropia-template-ui/style.css";
import "@xyflow/react/dist/style.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
