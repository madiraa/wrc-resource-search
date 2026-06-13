"""Data models for conversations, messages, and structured intake context."""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


NEED_TYPES = frozenset({
    "housing", "safety", "legal", "food", "childcare", "healthcare",
    "employment", "education", "financial", "other",
})

URGENCY_LEVELS = frozenset({"crisis", "this_week", " planning_ahead"})


@dataclass
class IntakeContext:
    """Structured memory extracted from conversation — drives matching and referral."""

    primary_need: Optional[str] = None
    urgency: Optional[str] = None
    need_summary: Optional[str] = None
    is_ccsf_student: Optional[str] = None  # yes / no / prefer_not_to_say
    housing_status: Optional[str] = None
    has_dependents: Optional[bool] = None
    deadline: Optional[str] = None
    language_preference: Optional[str] = None
    safe_contact_notes: Optional[str] = None
    additional_context: Optional[str] = None

    student_name: Optional[str] = None
    preferred_contact: Optional[str] = None  # phone / email / either
    student_phone: Optional[str] = None
    student_email: Optional[str] = None

    safety_review_required: bool = False
    safety_flags: list[str] = field(default_factory=list)
    crisis_detected: bool = False

    matched_resources: list[dict[str, Any]] = field(default_factory=list)
    selected_resource_id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntakeContext":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def completeness_for_matching(self) -> float:
        """Fraction of required matching fields populated."""
        required = ["primary_need", "urgency", "need_summary", "is_ccsf_student"]
        filled = sum(1 for f in required if getattr(self, f) not in (None, ""))
        return filled / len(required)

    def completeness_for_referral(self) -> float:
        """Fraction of fields needed to queue a referral."""
        match_score = self.completeness_for_matching()
        contact_fields = ["student_name", "preferred_contact"]
        contact_filled = sum(1 for f in contact_fields if getattr(self, f) not in (None, ""))
        if self.preferred_contact in ("phone", "either") and not self.student_phone:
            contact_filled -= 0.5
        if self.preferred_contact in ("email", "either") and not self.student_email:
            contact_filled -= 0.5
        contact_score = max(0.0, contact_filled / len(contact_fields))
        has_selection = 1.0 if self.selected_resource_id else 0.0
        return (match_score + contact_score + has_selection) / 3.0

    def to_intake_json(self) -> dict[str, Any]:
        """Backward-compatible dict for referral_queue.create_request()."""
        return {
            "primary_need": self.primary_need,
            "urgency": self.urgency,
            "is_ccsf_student": self.is_ccsf_student,
            "has_dependents": "yes" if self.has_dependents else "no",
            "housing_status": self.housing_status or "not applicable",
            "deadline": self.deadline or "",
            "additional_context": self.additional_context or "",
            "student_name": self.student_name or "",
            "student_phone": self.student_phone or "",
            "student_email": self.student_email or "",
            "preferred_contact": self.preferred_contact or "either",
            "safe_contact_notes": self.safe_contact_notes or "",
            "language_preference": self.language_preference or "",
        }


@dataclass
class Conversation:
    conversation_id: str
    created_at: str
    updated_at: str
    status: str  # active / abandoned / submitted / archived
    current_phase: str
    session_fingerprint: Optional[str] = None


@dataclass
class Message:
    message_id: str
    conversation_id: str
    role: str  # user / assistant / system
    content: str
    created_at: str
    phase: str
    metadata: Optional[dict[str, Any]] = None
