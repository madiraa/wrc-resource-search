# WRC Resource Search

Women's Resource Center resource finder for City College of San Francisco. A Streamlit app for searching support resources — with a planned next phase for guided intake and automated outreach to agencies on the student's behalf.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `wrc_resources.db` in the project root.

## Features today

- AI semantic search and keyword search over 800+ WRC resources
- Quick-search categories (housing, safety, legal, food, childcare, and more)
- CCSF and crisis resource prioritization
- Browse all resources and CSV export

## Next phase: intake & automated outreach

Instead of full case management (saved plans, staff dashboards), the next feature set focuses on:

1. **Asking structured questions** about the student's situation
2. **Matching** them to the best resources (existing RAG search)
3. **Reaching out to those resources on the student's behalf** — consent-based referral emails so the student isn't left to make the cold call alone

See **[PROJECT_PLAN.md](./PROJECT_PLAN.md)** for scope, safety rules, open questions for stakeholders, and phased implementation.

## Deploy (Streamlit Cloud)

1. Push this repo to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set main file to `app.py`

Email outreach (Phase 2) will require CCSF-approved SMTP or SendGrid credentials in Streamlit secrets.

## Project structure

```
app.py              # Streamlit web app
database.py         # SQLite schema and queries
rag_system.py       # Semantic search with embeddings
wrc_resources.db    # Resource database
PROJECT_PLAN.md     # Intake + outreach roadmap
```

## Notes

- Built for the Women's Resource Center at City College of San Francisco
- Resource contact details should be verified regularly; ~25% of current resources have email on file
- Automated outreach requires staff review for safety-related intakes (see project plan)
