# WRC Resource Search

Women's Resource Center resource finder for City College of San Francisco. A Streamlit app for searching support resources, with a guided intake flow (evolving into **AI chat**) and **mandatory staff approval** before any referral is sent.

**Technical scope for conversational intake:** [TECHNICAL_SCOPE_INTAKE.md](./TECHNICAL_SCOPE_INTAKE.md)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `wrc_resources.db` in the project root.

## App pages

| Page | Who | Purpose |
| --- | --- | --- |
| **Resource Search** (home) | Anyone | AI + keyword search over 800+ resources |
| **Get Help** | Students | Intake (form today → **AI chat** target) → consent → submit for review |
| **Staff Approval** | WRC staff (you) | Review student needs + matched org → approve or reject |

## Referral workflow

```
Student completes intake & consents
        ↓
Request queued (pending approval) — nothing is sent yet
        ↓
You open Staff Approval — see needs context, matched org, draft email
        ↓
Approve (send referral) or Reject (with reason)
```

**No referral is sent without your explicit approval.**

## Staff Approval setup

Protect the approval page with a password in Streamlit secrets (`.streamlit/secrets.toml`):

```toml
approver_password = "your-secure-password"
```

Or set environment variable `APPROVER_PASSWORD`.

Optional (Phase 2 — email sending on approve):

```toml
WRC_FROM_EMAIL = "wrc-referrals@ccsf.edu"
SMTP_HOST = "smtp.example.com"
SMTP_PORT = "587"
SMTP_USERNAME = "..."
SMTP_PASSWORD = "..."
```

## Deploy (Streamlit Cloud)

1. Push this repo to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set main file to `app.py`
4. Add `approver_password` (and SMTP vars when ready) in app secrets

## Project structure

```
app.py                  # Resource Search (home)
pages/
  2_Get_Help.py         # Student intake → pending approval
  3_Staff_Approval.py   # Staff review queue
intake/                 # Matching, templates, safety checks
referral_queue.py       # referral_requests database
database.py             # Resource SQLite schema
rag_system.py           # Semantic search
PROJECT_PLAN.md         # Full scope & phases
```

## Next steps

See **[PROJECT_PLAN.md](./PROJECT_PLAN.md)** for Phase 2 (email sending on approve) and open questions for CCSF IT / WRC policy.
