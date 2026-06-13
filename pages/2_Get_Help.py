"""Student intake — submit a referral request for staff approval."""

import streamlit as st
from pathlib import Path

from intake.matcher import match_resources
from intake.outreach import build_need_summary, render_referral_email
from intake.safety import intake_requires_extra_review, outreach_channel, resource_is_outreach_blocked
from rag_system import RAGSystem
from referral_queue import ReferralQueue
from ui_styles import WRC_CSS

st.set_page_config(
    page_title="Get Help | WRC",
    page_icon="💜",
    layout="wide",
)

st.markdown(WRC_CSS, unsafe_allow_html=True)

NEED_OPTIONS = {
    "housing": "Housing",
    "safety": "Safety / domestic violence",
    "legal": "Legal aid",
    "food": "Food assistance",
    "childcare": "Childcare",
    "healthcare": "Healthcare",
    "employment": "Employment",
    "education": "Education",
    "financial": "Financial assistance",
    "other": "Other",
}

URGENCY_OPTIONS = {
    "crisis": "Crisis — need help today",
    "this_week": "This week",
    "planning_ahead": "Planning ahead",
}


@st.cache_resource
def init_rag():
    if not Path("wrc_resources.db").exists():
        return None
    return RAGSystem("wrc_resources.db", use_local_embeddings=True)


@st.cache_resource
def init_queue():
    queue = ReferralQueue("wrc_resources.db")
    queue.ensure_tables()
    return queue


def reset_intake():
    for key in list(st.session_state.keys()):
        if key.startswith("intake_") or key in ("intake_step", "matched_resources", "selected_resource_id"):
            del st.session_state[key]
    st.session_state.intake_step = 1


rag = init_rag()
queue = init_queue()

if rag is None:
    st.error("Resource database not found.")
    st.stop()

if "intake_step" not in st.session_state:
    st.session_state.intake_step = 1

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Get Help</div>
        <div class="hero-subtitle">Tell us about your situation — a WRC staff member will review your match before we contact any organization</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="panel-note">
        After you consent, your request goes to a <strong>staff reviewer</strong> who checks that the matched organization
        is appropriate for your situation. The referral is only sent once approved.
    </div>
    """,
    unsafe_allow_html=True,
)

step = st.session_state.intake_step

if step == 1:
    st.subheader("Step 1 — About your situation")

    with st.form("intake_form"):
        col1, col2 = st.columns(2)

        with col1:
            primary_need = st.selectbox(
                "What do you need help with?",
                options=list(NEED_OPTIONS.keys()),
                format_func=lambda k: NEED_OPTIONS[k],
            )
            urgency = st.selectbox(
                "How urgent is this?",
                options=["crisis", "this_week", "planning_ahead"],
                format_func=lambda k: URGENCY_OPTIONS[k],
            )
            is_ccsf_student = st.selectbox(
                "Are you a CCSF student?",
                ["yes", "no", "prefer not to say"],
            )
            has_dependents = st.selectbox("Do you have dependents / children?", ["no", "yes"])

        with col2:
            housing_status = st.selectbox(
                "Housing situation",
                [
                    "not applicable",
                    "stable",
                    "temporary",
                    "at_risk_of_eviction",
                    "unhoused",
                ],
                format_func=lambda v: v.replace("_", " ").title(),
            )
            deadline = st.text_input("Any relevant deadline? (optional)")
            additional_context = st.text_area(
                "Anything else we should know? (optional)",
                max_chars=500,
                height=100,
            )

        st.markdown("**Your contact information** (for the referral, if approved)")
        c1, c2 = st.columns(2)
        with c1:
            student_name = st.text_input("Your name")
            student_phone = st.text_input("Phone (optional)")
        with c2:
            student_email = st.text_input("Email (optional)")
            preferred_contact = st.selectbox(
                "Preferred contact method for the agency",
                ["either", "phone", "email"],
            )

        safe_contact_notes = st.text_input(
            "Safe contact notes (optional)",
            placeholder='e.g. "Email only" or "Call after 5pm — do not leave voicemail"',
        )
        language_preference = st.text_input("Language preference (optional)")

        submitted = st.form_submit_button("Find matching resources", type="primary")

    if submitted:
        if not student_name.strip():
            st.error("Please enter your name so we can prepare a referral if approved.")
        elif preferred_contact in ("phone", "either") and not student_phone.strip():
            st.error("Please provide a phone number, or choose email as your preferred contact method.")
        elif preferred_contact in ("email", "either") and not student_email.strip():
            st.error("Please provide an email address, or choose phone as your preferred contact method.")
        else:
            intake = {
                "primary_need": primary_need,
                "urgency": urgency,
                "is_ccsf_student": is_ccsf_student,
                "has_dependents": has_dependents,
                "housing_status": housing_status,
                "deadline": deadline.strip(),
                "additional_context": additional_context.strip(),
                "student_name": student_name.strip(),
                "student_phone": student_phone.strip(),
                "student_email": student_email.strip(),
                "preferred_contact": preferred_contact,
                "safe_contact_notes": safe_contact_notes.strip(),
                "language_preference": language_preference.strip(),
            }
            st.session_state.intake_data = intake
            st.session_state.matched_resources = match_resources(rag, intake, top_k=3)
            st.session_state.intake_step = 2
            st.rerun()

elif step == 2:
    intake = st.session_state.get("intake_data", {})
    matches = st.session_state.get("matched_resources", [])

    if not matches:
        st.warning("No matching resources found. Try adjusting your answers or use Resource Search.")
        if st.button("Start over"):
            reset_intake()
            st.rerun()
        st.stop()

    st.subheader("Step 2 — Review your match")

    if intake_requires_extra_review(intake):
        st.warning(
            "Your situation may involve safety concerns. A staff member will carefully review this request "
            "before any outreach is sent."
        )

    selectable = [m for m in matches if not resource_is_outreach_blocked(m)]
    if not selectable:
        st.error(
            "The top matches are crisis hotlines or phone-only resources that cannot receive email referrals. "
            "Please visit the Resource Search page or speak with WRC staff directly."
        )
        if st.button("Start over"):
            reset_intake()
            st.rerun()
        st.stop()

    resource_labels = {
        r["resource_id"]: f"{r.get('organization_name', 'Unknown')} ({r.get('resource_type', 'general')})"
        for r in selectable
    }

    default_id = selectable[0]["resource_id"]
    selected_id = st.radio(
        "Select the organization you want us to contact on your behalf",
        options=list(resource_labels.keys()),
        format_func=lambda rid: resource_labels[rid],
        index=0,
    )
    selected = next(r for r in selectable if r["resource_id"] == selected_id)

    channel = outreach_channel(selected)
    if channel == "email":
        st.success(f"This organization can receive an email referral at: {selected.get('email')}")
    elif channel == "phone":
        st.info(
            f"This organization does not have email on file. If approved, WRC may call on your behalf "
            f"or provide a call script for: {selected.get('phone')}"
        )
    else:
        st.warning("This organization may require manual follow-up by WRC staff.")

    with st.expander("Why this resource?", expanded=True):
        st.write(selected.get("description") or "No description available.")
        if selected.get("eligibility"):
            st.markdown(f"**Eligibility:** {selected['eligibility']}")
        if selected.get("phone"):
            st.markdown(f"**Phone:** {selected['phone']}")
        if selected.get("hours"):
            st.markdown(f"**Hours:** {selected['hours']}")

    default_summary = build_need_summary(intake)
    need_summary = st.text_area(
        "Need summary (editable — this goes in the referral if approved)",
        value=default_summary,
        height=140,
    )

    subject, body = render_referral_email(intake, selected, need_summary)
    st.markdown("**Draft referral message (sent only after staff approval)**")
    st.text_input("Subject", value=subject, disabled=True)
    st.text_area("Message preview", value=body, height=320, disabled=True)

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back"):
            st.session_state.intake_step = 1
            st.rerun()
    with col_next:
        if st.button("Continue to consent →", type="primary"):
            st.session_state.selected_resource = selected
            st.session_state.need_summary = need_summary
            st.session_state.outreach_subject = subject
            st.session_state.outreach_body = body
            st.session_state.intake_step = 3
            st.rerun()

elif step == 3:
    intake = st.session_state.get("intake_data", {})
    selected = st.session_state.get("selected_resource")
    need_summary = st.session_state.get("need_summary", "")

    if not selected:
        st.session_state.intake_step = 2
        st.rerun()

    st.subheader("Step 3 — Consent & submit for review")

    st.markdown(
        f"""
        You selected **{selected.get('organization_name')}**.

        Your request will be reviewed by a WRC staff member before any referral is sent.
        """
    )

    consent_authorize = st.checkbox(
        "I authorize WRC to contact the organization above with the information I reviewed, if staff approve the match"
    )
    consent_understand = st.checkbox(
        "I understand WRC is making a referral, not guaranteeing services"
    )
    consent_accurate = st.checkbox("I have reviewed the message preview and it is accurate")

    col_back, col_submit = st.columns(2)
    with col_back:
        if st.button("← Back"):
            st.session_state.intake_step = 2
            st.rerun()
    with col_submit:
        if st.button("Submit for staff review", type="primary"):
            if not (consent_authorize and consent_understand and consent_accurate):
                st.error("Please confirm all consent items before submitting.")
            else:
                request_id = queue.create_request(
                    intake=intake,
                    matched_resources=[selected],
                    need_summary=need_summary,
                    outreach_subject=st.session_state.outreach_subject,
                    outreach_body=st.session_state.outreach_body,
                    safety_review_required=intake_requires_extra_review(intake),
                )
                st.session_state.submitted_request_id = request_id
                st.session_state.intake_step = 4
                st.rerun()

elif step == 4:
    request_id = st.session_state.get("submitted_request_id", "")
    st.subheader("Request submitted")
    st.markdown('<span class="status-pending">Pending staff review</span>', unsafe_allow_html=True)
    st.success(
        "Thank you. A WRC staff member will review your situation and the matched organization "
        "before any referral is sent."
    )
    st.markdown(
        f"""
        **Reference ID:** `{request_id[:8]}...`

        **What happens next**
        1. A staff reviewer sees your needs and the matched organization
        2. They confirm the match is appropriate
        3. If approved, WRC sends the referral on your behalf
        4. You should expect follow-up within 1–2 business days
        """
    )
    if st.button("Submit another request"):
        reset_intake()
        st.session_state.intake_step = 1
        st.rerun()
