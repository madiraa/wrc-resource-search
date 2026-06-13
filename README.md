# WRC Resource Search

Women's Resource Center resource finder for City College of San Francisco. This repository contains:

- **Streamlit app** (`app.py`) — full-featured search with AI semantic search via embeddings
- **Static GitHub Pages site** (`docs/`) — public, fast keyword search anyone can use without a server
- **Case management plan** (`PROJECT_PLAN.md`) — roadmap for personalized action plans and document export

## Live site

After GitHub Pages is enabled, the public site will be available at:

**https://madiraa.github.io/wrc-resource-search/**

## Why two versions?

GitHub Pages only hosts static HTML, CSS, and JavaScript. It cannot run Python, Streamlit, or the sentence-transformers embedding model. The static site solves public access; the Streamlit app keeps the richer AI search experience.

| Feature | GitHub Pages | Streamlit app |
| --- | --- | --- |
| Public URL | Yes | Requires Streamlit Cloud or server |
| Keyword / fuzzy search | Yes | Yes |
| AI semantic search | No | Yes |
| Browse & CSV download | Yes | Yes |
| Case plan builder (planned) | Phase 2 | Phase 2 |

## Run locally

### Streamlit app (full AI search)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `wrc_resources.db` in the project root.

### Static GitHub Pages preview

```bash
python scripts/export_resources.py
python -m http.server 8080 --directory docs
```

Open http://localhost:8080/wrc-resource-search/ if serving from repo root, or http://localhost:8080 when serving directly from `docs/`.

## Deploy to GitHub Pages

1. In the GitHub repo, go to **Settings → Pages**
2. Under **Build and deployment**, set **Source** to **GitHub Actions**
3. Push to `main` — the workflow in `.github/workflows/pages.yml` exports fresh JSON and deploys `docs/`

To redeploy manually: **Actions → Deploy GitHub Pages → Run workflow**

## Updating resource data

When the SQLite database changes:

```bash
python scripts/export_resources.py
git add docs/data/resources.json
git commit -m "Refresh exported resource data"
git push
```

The GitHub Actions workflow also regenerates the JSON on every push to `main`.

## Optional: Streamlit Cloud

For the AI-powered version:

1. Push this repo to GitHub
2. Connect it at [share.streamlit.io](https://share.streamlit.io)
3. Set the main file to `app.py`

Streamlit Cloud runs the Python backend and embedding model; GitHub Pages remains the lightweight public entry point.

## Next phase

See [PROJECT_PLAN.md](./PROJECT_PLAN.md) for the case-management system scope: situation intake, resource selection, action-plan generation, and downloadable documents.

## Project structure

```
app.py                 # Streamlit web app
database.py            # SQLite schema and queries
rag_system.py          # Semantic search with embeddings
wrc_resources.db       # Resource database
scripts/
  export_resources.py  # Export JSON for GitHub Pages
docs/
  index.html           # Static site
  css/styles.css
  js/app.js
  data/resources.json  # Generated search index
.github/workflows/
  pages.yml            # GitHub Pages deployment
PROJECT_PLAN.md        # Case management roadmap
```

## License & use

Built for the Women's Resource Center at City College of San Francisco. Resource data should be reviewed regularly for accuracy, especially contact details and eligibility requirements.
