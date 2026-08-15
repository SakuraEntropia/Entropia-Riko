// Entropia Riko — Electron shell (standalone desktop app).
//
// Spawns the FastAPI backend (if the bundled venv exists) and opens a window
// on the Vite dev server. Users who prefer the browser version keep using
// `npm run dev` and http://localhost:5173 directly.
const { app, BrowserWindow, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

let apiProcess = null;

function startApi() {
  const root = path.join(__dirname, "..");
  const python = process.platform === "win32"
    ? path.join(root, ".venv", "Scripts", "python.exe")
    : path.join(root, ".venv", "bin", "python");
  try {
    apiProcess = spawn(
      python,
      ["-m", "uvicorn", "src.server.app:app", "--port", "8000"],
      { cwd: root, stdio: "ignore" }
    );
  } catch {
    // If the venv is missing, assume the API is already running on :8000.
    apiProcess = null;
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: "Entropia Riko",
    backgroundColor: "#0d0f14",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const devUrl = process.env.RIKO_DEV_URL || "http://localhost:5173";
  win.loadURL(devUrl);

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
}

app.whenReady().then(() => {
  startApi();
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (apiProcess) apiProcess.kill();
  if (process.platform !== "darwin") app.quit();
});
