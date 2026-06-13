"""Send approved referral emails (Phase 2 — requires SMTP/SendGrid secrets)."""

import os
from typing import Optional


def send_referral_email(
    to_email: str,
    subject: str,
    body: str,
    reply_to: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Send a referral email. Returns (success, message).

    Configure via environment or Streamlit secrets:
      SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, WRC_FROM_EMAIL
    """
    from_email = os.getenv("WRC_FROM_EMAIL")
    if not from_email:
        return False, "Email not configured. Set WRC_FROM_EMAIL and SMTP credentials."

    smtp_host = os.getenv("SMTP_HOST")
    if not smtp_host:
        return False, "SMTP not configured. Referral approved but email was not sent."

    return False, "SMTP sending not yet implemented — referral marked approved for manual send."
