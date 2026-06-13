"""ChatEngine — orchestrates turn processing (Phase B implementation stub)."""

from typing import Any, Optional

from chat.models import IntakeContext
from chat.phases import IntakePhase
from chat.prompts import GREETING_OPENER, build_system_prompt
from chat.store import ConversationStore


class ChatEngine:
    """
    Processes one conversation turn: persist message, update context, produce reply.

    Phase B will add: LLM calls (Anthropic), structured extraction, RAG triggers,
    and phase transition rules. This stub supports storage + greeting only.
    """

    def __init__(
        self,
        store: Optional[ConversationStore] = None,
        rag=None,
    ):
        self.store = store or ConversationStore()
        self.store.ensure_tables()
        self.rag = rag

    def start_conversation(self, session_fingerprint: Optional[str] = None) -> dict[str, Any]:
        conversation = self.store.create_conversation(session_fingerprint=session_fingerprint)
        self.store.save_context_snapshot(conversation.conversation_id, IntakeContext())
        assistant_msg = self.store.add_message(
            conversation_id=conversation.conversation_id,
            role="assistant",
            content=GREETING_OPENER,
            phase=IntakePhase.GREETING.value,
        )
        return {
            "conversation_id": conversation.conversation_id,
            "phase": IntakePhase.GREETING.value,
            "message": assistant_msg,
            "context": IntakeContext(),
        }

    def process_turn(
        self,
        conversation_id: str,
        user_message: str,
    ) -> dict[str, Any]:
        """
        Handle one user message. Full LLM pipeline to be implemented in Phase B.

        Returns assistant reply metadata for the UI.
        """
        conversation = self.store.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Unknown conversation: {conversation_id}")

        phase = conversation.current_phase
        context = self.store.get_latest_context(conversation_id)

        user_msg = self.store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            phase=phase,
        )

        # Phase B: safety scan, extractor, phase transitions, RAG, LLM reply
        placeholder_reply = (
            "Thank you for sharing that. I'm still being set up to have a full conversation — "
            "for now, please use the structured Get Help form, or a staff member can assist you in person. "
            "(Chat engine Phase B will enable full dialogue here.)"
        )

        assistant_msg = self.store.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=placeholder_reply,
            phase=phase,
            metadata={"engine": "stub", "phase_b_pending": True},
        )

        self.store.save_context_snapshot(
            conversation_id,
            context,
            trigger_message_id=user_msg.message_id,
        )

        return {
            "conversation_id": conversation_id,
            "phase": phase,
            "user_message": user_msg,
            "assistant_message": assistant_msg,
            "context": context,
            "completeness_matching": context.completeness_for_matching(),
            "completeness_referral": context.completeness_for_referral(),
            "system_prompt_preview": build_system_prompt(phase, context.need_summary or ""),
        }
