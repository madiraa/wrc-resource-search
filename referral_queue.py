"""Referral request queue — pending staff approval before outreach is sent."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


STATUS_PENDING = "pending_approval"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_SENT = "sent"


class ReferralQueue:
    """Stores referral requests awaiting human approval."""

    def __init__(self, db_path: str = "wrc_resources.db"):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS referral_requests (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending_approval',
                    primary_need TEXT,
                    urgency TEXT,
                    intake_json TEXT NOT NULL,
                    need_summary TEXT,
                    student_name TEXT,
                    student_email TEXT,
                    student_phone TEXT,
                    preferred_contact TEXT,
                    safe_contact_notes TEXT,
                    matched_resources_json TEXT NOT NULL,
                    outreach_subject TEXT,
                    outreach_body TEXT,
                    safety_review_required INTEGER DEFAULT 0,
                    consent_at TEXT,
                    reviewed_at TEXT,
                    reviewed_by TEXT,
                    reviewer_notes TEXT,
                    rejection_reason TEXT,
                    sent_at TEXT,
                    sent_to_email TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_referral_status
                ON referral_requests(status)
                """
            )
            conn.commit()

    def create_request(
        self,
        intake: dict[str, Any],
        matched_resources: list[dict[str, Any]],
        need_summary: str,
        outreach_subject: str,
        outreach_body: str,
        safety_review_required: bool = False,
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        request_id = str(uuid.uuid4())
        primary_resource = matched_resources[0] if matched_resources else {}
        payload = {
            "request_id": request_id,
            "created_at": now,
            "updated_at": now,
            "status": STATUS_PENDING,
            "primary_need": intake.get("primary_need"),
            "urgency": intake.get("urgency"),
            "intake_json": json.dumps(intake),
            "need_summary": need_summary,
            "student_name": intake.get("student_name"),
            "student_email": intake.get("student_email"),
            "student_phone": intake.get("student_phone"),
            "preferred_contact": intake.get("preferred_contact"),
            "safe_contact_notes": intake.get("safe_contact_notes"),
            "matched_resources_json": json.dumps(matched_resources),
            "outreach_subject": outreach_subject,
            "outreach_body": outreach_body,
            "safety_review_required": 1 if safety_review_required else 0,
            "consent_at": now,
        }

        columns = ", ".join(payload.keys())
        placeholders = ", ".join("?" for _ in payload)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO referral_requests ({columns}) VALUES ({placeholders})",
                tuple(payload.values()),
            )
            conn.commit()

        return request_id

    def list_requests(
        self, status: Optional[str] = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM referral_requests"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._deserialize(row) for row in rows]

    def get_request(self, request_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM referral_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return self._deserialize(row) if row else None

    def approve_request(
        self,
        request_id: str,
        reviewed_by: str,
        reviewer_notes: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE referral_requests
                SET status = ?, updated_at = ?, reviewed_at = ?,
                    reviewed_by = ?, reviewer_notes = ?, rejection_reason = NULL
                WHERE request_id = ? AND status = 'pending_approval'
                """,
                (STATUS_APPROVED, now, now, reviewed_by, reviewer_notes, request_id),
            )
            conn.commit()

    def reject_request(
        self,
        request_id: str,
        reviewed_by: str,
        rejection_reason: str,
        reviewer_notes: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE referral_requests
                SET status = ?, updated_at = ?, reviewed_at = ?,
                    reviewed_by = ?, reviewer_notes = ?, rejection_reason =?
                WHERE request_id = ? AND status = 'pending_approval'
                """,
                (
                    STATUS_REJECTED,
                    now,
                    now,
                    reviewed_by,
                    reviewer_notes,
                    rejection_reason,
                    request_id,
                ),
            )
            conn.commit()

    def mark_sent(self, request_id: str, sent_to_email: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE referral_requests
                SET status = ?, updated_at = ?, sent_at = ?, sent_to_email =?
                WHERE request_id = ? AND status = 'approved'
                """,
                (STATUS_SENT, now, now, sent_to_email, request_id),
            )
            conn.commit()

    def pending_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM referral_requests WHERE status = ?",
                (STATUS_PENDING,),
            ).fetchone()
        return row["count"]

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["intake"] = json.loads(data.pop("intake_json"))
        data["matched_resources"] = json.loads(data.pop("matched_resources_json"))
        data["safety_review_required"] = bool(data.get("safety_review_required"))
        return data
