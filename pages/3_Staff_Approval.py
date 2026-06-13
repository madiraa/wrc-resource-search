"""Staff approval queue — review student needs and matched orgs before referral is sent."""

import os
import streamlit as st

from intake.email_sender import send_referral_email
from intake.safety import outreach_channel
from referral_queue import ReferralQueue, STATUS_APPROVED, STATUS_PENDING, STATUS_REJECTED, STATUS_SENT
from ui_styles import WRC_CSS

st.set_page_config(
    page_title="Staff Approval | WRC",
    page_icon="🔒",
    layout="wide",
)

st.markdown(WRC_CSS, unsafe_allow_html=True)


@st.cache_resource
def init_queue():
    queue = ReferralQueue("wrc_resources.db")
    queue.ensure_tables()
    return queue


def get_approver_password() -> str:
    try:
        return st.secrets.get("approver_password", "")
    except Exception:
        return os.getenv("APPROVER_PASSWORD", "")


def require_auth() -> bool:
    password = get_approver_password()
    if not password:
        st.warning(
            "Set `approver_password` in Streamlit secrets or `APPROVER_PASSWORD` in the environment "
            "to protect this page."
        )
        return True

    if st.session_state.get("staff_authenticated"):
        return True

    st.subheader("Staff sign-in")
    entered = st.text_input("Approver password", type="password")
    if st.button("Sign in", type="primary"):
        if entered == password:
            st.session_state.staff_authenticated = True
            st.session_state.reviewer_name = "WRC Staff"
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def status_badge(status: str) -> str:
    css = {
        STATUS_PENDING: "status-pending",
        STATUS_APPROVED: "status-approved",
        STATUS_REJECTED: "status-rejected",
        STATUS_SENT: "status-approved",
    }.get(status, "status-pending")
    label = status.replace("_", " ").title()
    return f'<span class="{css}">{label}</span>'


def render_intake_overview(request: dict) -> None:
    intake = request["intake"]
    st.markdown("### Student needs & context")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Primary need:** {request.get('primary_need', '—').replace('_', ' ')}")
        st.markdown(f"**Urgency:** {request.get('urgency', '—').replace('_', ' ')}")
        st.markdown(f"**CCSF student:** {intake.get('is_ccsf_student', '—')}")
        st.markdown(f"**Dependents:** {intake.get('has_dependents', '—')}")
        st.markdown(f"**Housing:** {intake.get('housing_status', '—').replace('_', ' ')}")
    with col2:
        st.markdown(f"**Student name:** {request.get('student_name') or '—'}")
        st.markdown(f"**Phone:** {request.get('student_phone') or '—'}")
        st.markdown(f"**Email:** {request.get('student_email') or '—'}")
        st.markdown(f"**Preferred contact:** {request.get('preferred_contact') or '—'}")
        if request.get("safe_contact_notes"):
            st.markdown(f"**Safe contact notes:** {request['safe_contact_notes']}")
        if intake.get("deadline"):
            st.markdown(f"**Deadline:** {intake['deadline']}")

    if intake.get("additional_context"):
        st.markdown(f"**Additional context:** {intake['additional_context']}")

    st.markdown(f"**Need summary (for referral):** {request.get('need_summary') or '—'}")

    if request.get("safety_review_required"):
        st.error("Safety review flagged — confirm this match carefully before approving outreach.")


def render_matched_orgs(request: dict) -> None:
    st.markdown("### Matched organization(s)")
    for resource in request.get("matched_resources", []):
        css = "match-card ccsf" if resource.get("is_ccsf") else "match-card"
        channel = outreach_channel(resource)
        st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
        st.markdown(f"**{resource.get('organization_name', 'Unknown')}**")
        st.markdown(f"Type: {resource.get('resource_type', 'general')} · Match score: {resource.get('similarity_score', 0):.0%}" if resource.get("similarity_score") else f"Type: {resource.get('resource_type', 'general')}")
        if resource.get("description"):
            st.caption(resource["description"][:300])
        contact_bits = []
        if resource.get("email"):
            contact_bits.append(f"Email: {resource['email']}")
        if resource.get("phone"):
            contact_bits.append(f"Phone: {resource['phone']}")
        if resource.get("eligibility"):
            contact_bits.append(f"Eligibility: {resource['eligibility'][:200]}")
        if contact_bits:
            st.markdown(" · ".join(contact_bits))
        st.markdown(f"Outreach channel if approved: **{channel}**")
        st.markdown("</div>", unsafe_allow_html=True)


def render_outreach_preview(request: dict) -> None:
    st.markdown("### Referral message preview")
    st.text_input("Subject", value=request.get("outreach_subject") or "", disabled=True)
    st.text_area("Body", value=request.get("outreach_body") or "", height=280, disabled=True)


queue = init_queue()

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Staff Approval Queue</div>
        <div class="hero-subtitle">Review student needs and matched organizations before any referral is sent</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not require_auth():
    st.stop()

pending_count = queue.pending_count()
st.metric("Pending review", pending_count)

if st.session_state.get("staff_authenticated") and get_approver_password():
    if st.button("Sign out"):
        st.session_state.staff_authenticated = False
        st.rerun()

reviewer = st.session_state.get("reviewer_name", "WRC Staff")

tab_pending, tab_history = st.tabs(["Pending approval", "Recent decisions"])

with tab_pending:
    pending = queue.list_requests(status=STATUS_PENDING)
    if not pending:
        st.info("No referral requests waiting for review.")
    else:
        labels = {
            r["request_id"]: (
                f"{r.get('student_name', 'Student')} — "
                f"{r.get('primary_need', '').replace('_', ' ')} — "
                f"{r['created_at'][:10]}"
            )
            for r in pending
        }
        selected_id = st.selectbox(
            "Select a request to review",
            options=list(labels.keys()),
            format_func=lambda rid: labels[rid],
        )
        request = queue.get_request(selected_id)

        if request:
            st.markdown(status_badge(request["status"]), unsafe_allow_html=True)
            st.caption(f"Request ID: `{request['request_id']}` · Submitted: {request['created_at']}")

            render_intake_overview(request)
            render_matched_orgs(request)
            render_outreach_preview(request)

            st.markdown("### Your decision")
            reviewer_notes = st.text_area("Internal notes (optional)", height=80)
            rejection_reason = st.text_input(
                "Rejection reason (required if rejecting)",
                placeholder="e.g. Better match available — food pantry vs housing",
            )

            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button("Approve match & send referral", type="primary"):
                    primary = request["matched_resources"][0] if request["matched_resources"] else {}
                    channel = outreach_channel(primary)
                    queue.approve_request(selected_id, reviewer, reviewer_notes)

                    if channel == "email" and primary.get("email"):
                        success, message = send_referral_email(
                            to_email=primary["email"],
                            subject=request["outreach_subject"],
                            body=request["outreach_body"],
                        )
                        if success:
                            queue.mark_sent(selected_id, primary["email"])
                            st.success(f"Approved and referral sent to {primary['email']}.")
                        else:
                            st.warning(
                                f"Match approved, but email was not sent: {message} "
                                "You can send the previewed message manually."
                            )
                    else:
                        st.success(
                            "Match approved. This resource requires phone or manual outreach — "
                            "use the contact details above."
                        )
                    st.rerun()

            with col_reject:
                if st.button("Reject match"):
                    if not rejection_reason.strip():
                        st.error("Please provide a rejection reason.")
                    else:
                        queue.reject_request(
                            selected_id,
                            reviewer,
                            rejection_reason.strip(),
                            reviewer_notes,
                        )
                        st.success("Request rejected. Student may submit a new request.")
                        st.rerun()

with tab_history:
    history = queue.list_requests(limit=30)
    history = [r for r in history if r["status"] != STATUS_PENDING]
    if not history:
        st.info("No reviewed requests yet.")
    else:
        for item in history:
            org = (
                item["matched_resources"][0].get("organization_name")
                if item.get("matched_resources")
                else "—"
            )
            st.markdown(
                f"{status_badge(item['status'])} · **{item.get('student_name', 'Student')}** · "
                f"{item.get('primary_need', '').replace('_', ' ')} → {org} · "
                f"{item.get('reviewed_at') or item.get('created_at')}",
                unsafe_allow_html=True,
            )
            if item.get("rejection_reason"):
                st.caption(f"Rejected: {item['rejection_reason']}")
            if item.get("reviewer_notes"):
                st.caption(f"Notes: {item['reviewer_notes']}")
