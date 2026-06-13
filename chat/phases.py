"""Intake phase state machine for conversational intake."""

from enum import Enum


class IntakePhase(str, Enum):
    GREETING = "greeting"
    EXPLORING_NEED = "exploring_need"
    SAFETY_CHECK = "safety_check"
    CRISIS_RESOURCES = "crisis_resources"
    GATHERING_DETAILS = "gathering_details"
    MATCHING = "matching"
    PRESENTING_MATCHES = "presenting_matches"
    CONTACT_COLLECTION = "contact_collection"
    REFERRAL_PREVIEW = "referral_preview"
    CONSENT = "consent"
    PENDING_APPROVAL = "pending_approval"


# Phases where RAG search is permitted
RAG_PHASES = frozenset({
    IntakePhase.SAFETY_CHECK,
    IntakePhase.CRISIS_RESOURCES,
    IntakePhase.MATCHING,
    IntakePhase.PRESENTING_MATCHES,
})

# Terminal phases — no further student chat expected
TERMINAL_PHASES = frozenset({
    IntakePhase.CRISIS_RESOURCES,
    IntakePhase.PENDING_APPROVAL,
})
