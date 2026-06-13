"""Referral email template generation."""

from typing import Any


def build_search_query(intake: dict[str, Any]) -> str:
    """Turn intake answers into a RAG search query."""
    parts = [
        intake.get("primary_need", ""),
        intake.get("urgency", ""),
        intake.get("housing_status", ""),
        intake.get("additional_context", ""),
    ]
    if intake.get("has_dependents") == "yes":
        parts.append("with children or dependents")
    if intake.get("is_ccsf_student") == "yes":
        parts.append("CCSF student")
    return " ".join(part for part in parts if part).strip()


def build_need_summary(intake: dict[str, Any]) -> str:
    """Default need summary for the referral email."""
    need = intake.get("primary_need", "support").replace("_", " ")
    urgency = intake.get("urgency", "planning ahead")
    lines = [f"The student is seeking help with {need} (urgency: {urgency})."]

    if intake.get("housing_status"):
        lines.append(f"Housing situation: {intake['housing_status'].replace('_', ' ')}.")
    if intake.get("has_dependents") == "yes":
        lines.append("The student has dependents.")
    if intake.get("deadline"):
        lines.append(f"Relevant deadline: {intake['deadline']}.")
    if intake.get("additional_context"):
        lines.append(intake["additional_context"])

    return " ".join(lines)


def render_referral_email(
    intake: dict[str, Any],
    resource: dict[str, Any],
    need_summary: str,
) -> tuple[str, str]:
    """Return subject and body for the referral email."""
    need_label = intake.get("primary_need", "support").replace("_", " ")
    org_name = resource.get("organization_name") or "your organization"
    subject = f"CCSF Women's Resource Center referral — {need_label} support requested"

    preferred = intake.get("preferred_contact") or "either"
    contact_lines = []
    if intake.get("student_name"):
        contact_lines.append(f"  Name: {intake['student_name']}")
    if preferred in ("phone", "either") and intake.get("student_phone"):
        contact_lines.append(f"  Phone: {intake['student_phone']}")
    if preferred in ("email", "either") and intake.get("student_email"):
        contact_lines.append(f"  Email: {intake['student_email']}")
    if intake.get("safe_contact_notes"):
        contact_lines.append(f"  Contact notes: {intake['safe_contact_notes']}")
    if intake.get("language_preference"):
        contact_lines.append(f"  Language preference: {intake['language_preference']}")

    contact_block = "\n".join(contact_lines) if contact_lines else "  (Contact details provided separately by WRC.)"

    student_status = intake.get("is_ccsf_student", "prefer not to say")
    body = f"""Hello {org_name},

The Women's Resource Center at City College of San Francisco is referring a student who has asked for help with {need_label}.

The student has authorized us to share the following:

{contact_block}

  Need summary: {need_summary}
  Urgency: {intake.get('urgency', 'not specified')}
  CCSF student: {student_status}

The student is aware you may follow up directly. Please reply to this email or contact the student using the method above.

If your program is not accepting referrals or the student is not eligible, a brief reply helps us redirect them.

Thank you for the work you do in our community.

Women's Resource Center, City College of San Francisco
"""

    return subject, body.strip()
