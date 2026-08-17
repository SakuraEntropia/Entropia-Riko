import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./theme"; // initialize light/dark/system theme
import "./styles.css";
import "@xyflow/react/dist/style.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
