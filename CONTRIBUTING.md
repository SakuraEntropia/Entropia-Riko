# Contributing

Thanks for helping maintain Entropia Riko. This file gets you from clone to a
merged change with the least friction.

## Quickstart

```bash
git clone https://github.com/SakuraEntropia/Entropia-Riko.git
cd Entropia-Riko

# Python backend
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend
npm install
```

The backend API runs on `:8000`, the Vite dev server on `:5173`:

```bash
.venv/bin/python -m uvicorn entropia_riko.server.app:app --reload --port 8000
npm run dev
```

## Development loop

```bash
# Backend tests (unittest)
.venv/bin/python -m unittest discover -s tests -t .

# Frontend type-check + build
npm run build

# Frontend tests (vitest)
npm test
```

## Code style

Python is linted and auto-fixed with [ruff](https://docs.astral.sh/ruff/):

```bash
.venv/bin/python -m pip install ruff
.venv/bin/python -m ruff check .            # report issues
.venv/bin/python -m ruff check . --fix      # auto-fix
.venv/bin/python -m ruff format .           # format
```

Rules live in `[tool.ruff]` in `pyproject.toml`. Keep the linter clean before
committing — CI runs `ruff check` on every push.

TypeScript is checked by `tsc` as part of `npm run build`.

## How changes are structured

- **Add a node** → subclass `BaseNode`, `@register("type_name")`, import it in
  `entropia_riko/nodes/__init__.py`, add a test in `tests/`. See
  `ARCHITECTURE.md#adding-a-node`.
- **Change the API** → edit the router in `entropia_riko/server/routers/`,
  include it in `server/app.py`, add a test in `tests/test_api.py`.
- **Change the UI** → edit `entropia_riko/ui/` (React + zustand). Keep the
  `ui/` tree in sync with the `Entropia-Template-UI` repo if the change is
  reusable.

## Commit & pull request

1. One concern per commit; write a clear imperative summary
   (e.g. `feat: add train-to-model node`).
2. Run the backend tests and `npm run build` before pushing.
3. Open a PR against `main`; CI runs tests, build, and lint automatically.

## Releasing

Releases are versioned `0.x.y`. To cut one:

1. Bump `version` in `pyproject.toml`, `entropia_riko/__init__.py`, and
   `package.json`.
2. `.venv/bin/python scripts/release.py "release note" --push` builds
   `entropia-riko-release.zip` and pushes.
3. Tag + publish the GitHub release, then upload to PyPI / npm:

```bash
git tag v0.x.y && git push origin v0.x.y
gh release create v0.x.y ../entropia-riko-release.zip --title "Entropia Riko v0.x.y"

.venv/bin/python -m pip install build twine
.venv/bin/python -m build --outdir dist-pypi
.venv/bin/python -m twine upload dist-pypi/*
```

Use short-lived scoped tokens for PyPI/npm and revoke them after publishing.

## Getting help

- Architecture: `ARCHITECTURE.md`
- User-facing behavior: `docs/USER_GUIDE.md`
- API reference: `docs/API.md`
