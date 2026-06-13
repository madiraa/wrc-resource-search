"""Safety checks before a referral can be submitted or sent."""

from typing import Any


SAFETY_KEYWORDS = (
    "domestic violence",
    "partner monitoring",
    "abuser",
    "unsafe at home",
    "sexual assault",
    "stalking",
)


def intake_requires_extra_review(intake: dict[str, Any]) -> bool:
    """Return True when staff must carefully review before any outreach."""
    if intake.get("primary_need") == "safety":
        return True
    if intake.get("urgency") == "crisis":
        return True

    combined = " ".join(
        str(intake.get(key, ""))
        for key in ("additional_context", "safe_contact_notes", "need_summary")
    ).lower()
    return any(keyword in combined for keyword in SAFETY_KEYWORDS)


def resource_is_outreach_blocked(resource: dict[str, Any]) -> bool:
    """Hotlines and similar resources should never receive email referrals."""
    org = (resource.get("organization_name") or "").lower()
    blocked_names = (
        "san francisco women against rape",
        "sfwar",
    )
    return any(name in org for name in blocked_names)


def outreach_channel(resource: dict[str, Any]) -> str:
    """How outreach can happen for this resource."""
    if resource_is_outreach_blocked(resource):
        return "blocked"
    if resource.get("email"):
        return "email"
    if resource.get("phone"):
        return "phone"
    return "manual"
