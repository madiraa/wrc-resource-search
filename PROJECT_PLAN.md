# WRC Intake & Automated Resource Outreach — Project Plan & Scope

## Executive summary

The WRC Resource Search tool helps students **find** support. The next phase keeps everything in **Streamlit** and shifts away from full case-management (saved plans, staff dashboards, document exports).

Instead, the system will:

1. **Discuss needs in a conversational intake** with an AI assistant (replacing the interim form)
2. **Match** them to the best resources from `wrc_resources.db` (using existing RAG search)
3. **Reach out to those resources on the student’s behalf** — starting with a consent-based referral message — so the student is not left to make the first contact alone

**Technical scope for chat + context storage:** [TECHNICAL_SCOPE_INTAKE.md](./TECHNICAL_SCOPE_INTAKE.md)

This document defines scope, open questions, phases, and safety constraints so the outreach feature can be designed responsibly before any code is written.

---

## Why this direction instead of case management?

| Case management (previous plan) | Intake + outreach (this plan) |
| --- | --- |
| Student downloads a plan and calls agencies themselves | WRC/system initiates contact with the agency |
| More tracking, storage, and staff workflow | Lighter footprint; focus on removing the “first call” barrier |
| Student still does the outreach | Student gets a warm handoff: “Someone is expecting you” |

Many students stall at the **first contact** — not knowing what to say, fear of rejection, language barriers, or trauma. Automating (or semi-automating) that first outreach can be more impactful than giving them another PDF.

---

## Problem statement

Students leave the WRC with resource names but still face:

- **Cold-call anxiety** — “What do I even say?”
- **Eligibility uncertainty** — they don’t know if they qualify before calling
- **Capacity limits** — students may not have time or privacy to call during agency hours
- **Follow-through gap** — especially for housing, legal, and childcare where the first contact matters

Staff cannot make every referral call during drop-in hours. A guided intake that produces a **consented, templated outreach** to the right resource extends WRC support while keeping humans in the loop where safety requires it.

---

## Core user journey

**Every referral requires human approval.** No message is sent to an outside organization until a staff reviewer (you) assesses the student’s needs, the matched organization(s), and explicitly approves the send.

```mermaid
flowchart TD
    A[Student opens Get Help] --> B[AI intake chat — multi-turn]
    B --> C[Context stored — IntakeContext + transcript]
    C --> D[System matches 1–3 resources]
    D --> E[Student reviews match + outreach preview]
    E --> F[Student consent]
    F --> G[Request queued — pending approval]
    G --> H[Staff reviewer sees context + chat summary]
    H --> I{Suitable match?}
    I -->|No| J[Reject with reason]
    I -->|Yes| K[Approve → send referral]
    K --> L[Student confirmation]
```

### Mandatory staff approval gate

After the student consents, the system **does not send anything**. Instead it creates a `referral_request` with:

| What you see | Purpose |
| --- | --- |
| Student needs & context | Primary need, urgency, housing, dependents, deadlines, free text |
| Contact preferences | Name, phone/email, safe contact notes, language |
| Matched organization(s) | Name, type, contact info, eligibility, match score, outreach channel |
| Draft referral email | Exact message that would be sent |
| Safety flags | Crisis/DV-related intakes flagged for extra care |

**Your actions:**

- **Approve match & send referral** — confirms the org is appropriate; triggers email if available, or flags phone/manual follow-up
- **Reject match** — requires a reason (e.g., wrong org, need more info, safety concern)

Implemented in Streamlit: **Staff Approval** page (`pages/3_Staff_Approval.py`), password-protected via `approver_password` secret.

---

## Intake design — conversational (target)

The interim **form** in `pages/2_Get_Help.py` will be replaced by a **chatbot** that discusses the student's needs with AI, stores conversation context, and extracts structured fields for matching and referral.

See **[TECHNICAL_SCOPE_INTAKE.md](./TECHNICAL_SCOPE_INTAKE.md)** for:

- Chat infrastructure (`ChatEngine`, phased prompts, Anthropic LLM)
- Context storage (`conversations`, `messages`, `intake_context_snapshots` tables)
- Intake phase state machine (greeting → exploring → matching → consent)
- How chat handoff feeds staff approval (summary + transcript excerpt)

### Structured fields still required (extracted from chat)

The chat must still populate the same fields the form collects today — matching and referral depend on them:

| Category | Fields |
| --- | --- |
| Need | `primary_need`, `urgency`, `need_summary` |
| Context | CCSF student, dependents, housing status, deadline |
| Contact | name, phone/email, preferred method, safe contact notes |
| Safety | `safety_review_required`, crisis flags |

### Interim form (being replaced)

The form below remains until Phase C (chat UI) ships:

- What do you need help with? (housing, safety, legal, food, childcare, health, employment, education, financial, other)
- How urgent is this? (crisis / this week / planning ahead)

### Section 2 — Situation context

- Are you a CCSF student? (yes / no / prefer not to say)
- Do you have dependents / children?
- Housing status (stable, temporary, unhoused, at risk of eviction)
- Employment status (optional)
- Any deadlines? (e.g., court date, lease end, school start)

### Section 3 — Outreach preferences

- **Preferred contact method for the agency to reach you:** phone / email / either
- **Best time to reach you** (optional)
- **Safe contact note:** “Only call after 5pm” / “Do not leave voicemail” / “Email only”
- **Language preference** for follow-up

### Section 4 — Consent (required before send)

- [ ] I authorize WRC to contact the organization(s) below with the information I reviewed
- [ ] I understand WRC is making a **referral**, not guaranteeing services
- [ ] I have reviewed the message preview and it is accurate
- [ ] For safety intakes: additional acknowledgment if partner monitoring is a concern

### Section 5 — Free text (optional, capped)

- “Anything else we should include?” (500 char limit; warn that this goes to the agency)

**Design principle:** Every field should map to either **resource matching** or **outreach template slots**. No “nice to have” questions that don’t change outcomes.

---

## Resource matching (reuse existing stack)

Matching builds on what already exists:

| Component | Role |
| --- | --- |
| `rag_system.py` | Semantic search from intake summary |
| Priority tiers in RAG | CCSF reporting, crisis hotlines, campus resources first when relevant |
| `database.py` | Contact info, eligibility, hours |

**Intake → search query:** Combine primary need + urgency + 1–2 context fields into a natural-language query (rule-based or LLM-assisted).

**Output:** Top 1–3 resources, ranked. Student confirms which to contact (default: top match only; optional second choice if first is waitlisted).

### Data reality check

Current database coverage (approximate):

| Contact method | Current resources |
| --- | --- |
| Has email | ~25% (~188 of 756) |
| Has phone | ~57% (~434 of 756) |

**Implication:** Email outreach alone will not cover all resources. The plan must handle:

- **Email available** → automated/semi-automated send
- **Phone only** → generate a **call script + click-to-call** for student *or* queue for staff to call
- **No contact** → flag for WRC staff manual follow-up

This should be surfaced honestly in the UI: *“This organization can’t receive automated email — here’s how we’ll help instead.”*

---

## Outreach message template (draft)

```
Subject: CCSF Women's Resource Center referral — [Need type] support requested

Hello [Organization name],

The Women's Resource Center at City College of San Francisco is referring a 
student who has asked for help with [primary need].

Student has authorized us to share the following:

  Name: [Student name — or first name only, per policy]
  Preferred contact: [phone/email] — [safe contact notes]
  Need summary: [2–4 sentences from intake, student-approved]
  Urgency: [crisis / this week / planning]
  CCSF student: [yes/no if shared]

The student is aware you may follow up directly. Please reply to this email 
or contact the student using the method above.

If your program is not accepting referrals or the student is not eligible, 
a brief reply helps us redirect them.

Thank you for the work you do in our community.

[Staff name / WRC generic signature]
Women's Resource Center, City College of San Francisco
[Phone / room / hours]
```

**Student sees full preview and can edit the “Need summary” block before send.**

---

## Safety & privacy rules (non-negotiable)

Women’s Resource Center work intersects **FERPA**, **Title IX**, and **survivor safety**.

| Scenario | Rule |
| --- | --- |
| Domestic violence / partner monitoring | **No auto-send.** Show crisis resources; staff-only outreach or student-initiated contact |
| Sexual assault / Title IX | Surface Title IX first; outreach may require staff review |
| Student under 18 | Staff review required |
| Student declines to share name | Send referral with first name only or “WRC student” + contact method, per policy |
| Wrong email in database | Confirm email domain looks valid; flag low-confidence contacts |
| Agency is a hotline (SFWAR, etc.) | **Never email a hotline for a student.** Show number; student calls or staff assists in person |

**Default posture:** **Human approval required for every referral.** There is no auto-send path in the MVP. Safety-flagged intakes display an extra warning on the staff review screen.

---

## Architecture (Streamlit-only)

```mermaid
flowchart LR
    subgraph student [Student pages]
        INT[Get Help intake]
        CONSENT[Consent + submit]
    end

    subgraph staff [Staff page]
        QUEUE[Approval queue]
        REVIEW[Needs + match overview]
        DECIDE[Approve or reject]
    end

    subgraph backend [Backend]
        RAG[rag_system.py]
        DB[(wrc_resources.db)]
        RQ[referral_requests table]
        MAIL[Email SMTP / SendGrid]
    end

    INT --> RAG
    RAG --> DB
    CONSENT --> RQ
    QUEUE --> RQ
    REVIEW --> DECIDE
    DECIDE -->|approve| MAIL
    DECIDE --> RQ
```

### Module layout (implemented / planned)

```
chat/                       # NEW — conversational intake
  store.py                  # conversations, messages, context snapshots
  engine.py                 # ChatEngine (Phase B: full LLM pipeline)
  models.py                 # IntakeContext, Message, Conversation
  prompts.py                # Phase-specific system prompts
intake/
  matcher.py                # Intake → RAG query → ranked resources
  outreach.py               # Referral email template
  safety.py                 # Safety flags, outreach channel checks
  email_sender.py           # SMTP send (Phase 2)
referral_queue.py           # referral_requests CRUD + approval states
pages/
  2_Get_Help.py             # Student intake (form → chat in Phase C)
  3_Staff_Approval.py       # Staff review queue (password protected)
app.py                      # Resource Search (home page)
TECHNICAL_SCOPE_INTAKE.md   # Chat + context storage technical spec
```

---

## Functional requirements

### Phase 1 — Intake + staff approval queue ✅ (form interim)

- [x] Multi-step intake **form** in Streamlit (`pages/2_Get_Help.py`) — interim until chat UI
- [x] Intake → RAG query generation
- [x] Student selects match, edits need summary, previews referral
- [x] Consent → submit for **pending approval** (no send)
- [x] Staff approval page with needs overview, matched orgs, message preview
- [x] Approve / reject actions with notes
- [x] `referral_requests` table
- [x] Safety flags on review screen
- [x] Chat storage foundation (`chat/store.py`, `TECHNICAL_SCOPE_INTAKE.md`)

**Acceptance:** Student submits request → appears in Staff Approval queue → you approve or reject before any outreach.

### Phase 1b — Conversational intake (chat)

- [ ] `ChatEngine` with Anthropic LLM + phase state machine
- [ ] Structured field extraction from dialogue
- [ ] Chat UI replacing form in `pages/2_Get_Help.py`
- [ ] Conversation summary + transcript on Staff Approval page
- [ ] `conversation_id` linked on referral submit

**Acceptance:** Student completes intake via chat; context persisted; staff sees summary + can approve/reject.

### Phase 2 — Email sending on approval

- [ ] Configure WRC SMTP / SendGrid in Streamlit secrets
- [ ] Send referral email when staff approves and org has email
- [ ] Phone-only fallback workflow after approval
- [ ] Optional email notification to approver when new request is submitted

**Acceptance:** End-to-end referral email sent only after staff approval.

### Phase 3 — Student notifications & iteration

- [ ] Email copy to student (“We sent this on your behalf”)
- [ ] Optional: inbound reply parsing (“Agency said they’ll call you Tuesday”)
- [ ] Template library by need type (housing, legal, childcare)
- [ ] Track which resources respond / bounce (operational, not student-facing CRM)

### Phase 4 — Scale (only after review)

- [ ] Email notification to approver on new submissions
- [ ] CCSF SSO for staff page
- [ ] Multilingual intake and outreach templates

---

## Open questions for WRC / CCSF stakeholders

These should be answered **before** Phase 2 goes live:

### Outreach mechanics

1. **Sending identity:** Should emails come from a shared WRC inbox, a staff member’s address, or `noreply@` with reply-to set to WRC?
2. **Student identity:** Full name, first name only, or student ID?
3. **Student contact in referral:** Include student phone/email directly, or ask agency to contact WRC first?
4. **Staff approval:** **Every referral** — you review needs + match before send ✅
5. **Volume:** How many automated referrals per week can partner agencies realistically absorb?

### Legal & compliance

6. Does CCSF IT allow **SendGrid / SMTP** from Streamlit Cloud, or must this run on-campus?
7. Is this **FERPA** “directory information” or does it require separate consent forms?
8. Do any partner agencies **prohibit** third-party referrals by email?
9. Retention: How long do we keep `outreach_log` records?

### Safety

10. For DV intakes, is **any** automated message to an outside agency acceptable, or always in-person/staff call?
11. Should the app include a **quick exit** and session clear (standard for survivor tools)?
12. What happens if the student’s abuser monitors email — is email outreach ever allowed?

### Operational

13. Who monitors the WRC inbox for agency replies?
14. What is the **SLA** for staff to approve queued referrals (same day? 24h?)?
15. When email is missing (~75% of resources), is the fallback **staff phone call** or **student call script**?
16. Should we **verify** agency emails periodically (many binder entries may be stale)?

---

## Success metrics

| Metric | Target |
| --- | --- |
| Intake completion rate | >60% of students who start intake |
| Match relevance (staff spot-check) | >80% “appropriate resource” |
| Outreach sent within 24h of intake | >90% (with staff approval model) |
| Student follow-through | Track whether student reports agency contact (optional survey) |
| Safety incidents from auto-outreach | Zero — any breach triggers immediate manual-review-only mode |

---

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Stale/wrong emails in database | Validate format; staff review; periodic verification |
| Only ~25% of resources have email | Phone fallback + prioritize email-capable orgs in matching when outreach is goal |
| Student PII sent inappropriately | Preview + consent; minimal fields; staff approval |
| Agency overload / spam perception | Rate limits; one referral per student per need; clear WRC signature |
| Streamlit Cloud email restrictions | Confirm with CCSF IT early; may need on-prem or serverless function |
| Survivor safety | Hard blocks on auto-send for flagged intakes |

---

## What we are explicitly not building

- GitHub Pages / static site (Streamlit only)
- Full case-management CRM with notes, assignments, and document vaults
- AI phone calls to agencies
- Guaranteed placement or eligibility determination
- Long-term storage of sensitive narratives without policy sign-off

---

## Recommended next steps

1. **Review this plan with WRC leadership** — especially the open questions above
2. **Answer sending identity and consent policy** with CCSF legal/IT
3. **Phase 1 implementation:** intake wizard + match preview in Streamlit (no email yet)
4. **Pilot with 2–3 email-friendly partners** (e.g., food, housing, childcare orgs with known inboxes)
5. **Measure staff time saved** vs. students making cold calls themselves

---

## Summary

The next evolution of the WRC Resource Search app is a **Streamlit intake assistant** that understands the student’s needs, finds the right resource, and **makes the first contact for them** — with clear consent, safety guardrails, and staff oversight where it matters. That removes the hardest step for many students while staying lighter than a full case-management system.

When you’re ready to build, Phase 1 (intake + match + outreach preview, no send) is the right place to start — it delivers value immediately and forces the stakeholder conversations this feature requires.
