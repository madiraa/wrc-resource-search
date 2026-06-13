"""SQLite persistence for conversations, messages, and intake context snapshots."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from chat.models import Conversation, IntakeContext, Message
from chat.phases import IntakePhase


class ConversationStore:
    """CRUD for chat infrastructure tables."""

    def __init__(self, db_path: str = "wrc_resources.db"):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    current_phase TEXT NOT NULL DEFAULT 'greeting',
                    session_fingerprint TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, created_at);

                CREATE TABLE IF NOT EXISTS intake_context_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    completeness_matching REAL DEFAULT 0,
                    completeness_referral REAL DEFAULT 0,
                    trigger_message_id TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                );

                CREATE INDEX IF NOT EXISTS idx_context_conversation
                ON intake_context_snapshots(conversation_id, created_at);
                """
            )
            # Extend referral_requests with conversation link if missing
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(referral_requests)").fetchall()
            }
            if cols and "conversation_id" not in cols:
                conn.execute("ALTER TABLE referral_requests ADD COLUMN conversation_id TEXT")
            if cols and "conversation_summary" not in cols:
                conn.execute("ALTER TABLE referral_requests ADD COLUMN conversation_summary TEXT")
            if cols and "transcript_excerpt" not in cols:
                conn.execute("ALTER TABLE referral_requests ADD COLUMN transcript_excerpt TEXT")
            conn.commit()

    def create_conversation(
        self,
        session_fingerprint: Optional[str] = None,
        phase: IntakePhase = IntakePhase.GREETING,
    ) -> Conversation:
        now = datetime.now(timezone.utc).isoformat()
        conversation_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations
                (conversation_id, created_at, updated_at, status, current_phase, session_fingerprint)
                VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (conversation_id, now, now, phase.value, session_fingerprint),
            )
            conn.commit()
        return Conversation(
            conversation_id=conversation_id,
            created_at=now,
            updated_at=now,
            status="active",
            current_phase=phase.value,
            session_fingerprint=session_fingerprint,
        )

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if not row:
            return None
        return Conversation(**dict(row))

    def update_phase(self, conversation_id: str, phase: IntakePhase) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET current_phase = ?, updated_at = ?, status = 'active'
                WHERE conversation_id =?
                """,
                (phase.value, now, conversation_id),
            )
            conn.commit()

    def update_status(self, conversation_id: str, status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET status = ?, updated_at = ? WHERE conversation_id = ?",
                (status, now, conversation_id),
            )
            conn.commit()

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        phase: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        now = datetime.now(timezone.utc).isoformat()
        message_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata) if metadata else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages
                (message_id, conversation_id, role, content, created_at, phase, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content, now, phase, metadata_json),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
            conn.commit()
        return Message(
            message_id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
            phase=phase,
            metadata=metadata,
        )

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> list[Message]:
        query = """
            SELECT * FROM messages
            WHERE conversation_id =?
            ORDER BY created_at ASC
        """
        params: list[Any] = [conversation_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        messages = []
        for row in rows:
            data = dict(row)
            meta = data.pop("metadata_json", None)
            data["metadata"] = json.loads(meta) if meta else None
            messages.append(Message(**data))
        return messages

    def save_context_snapshot(
        self,
        conversation_id: str,
        context: IntakeContext,
        trigger_message_id: Optional[str] = None,
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        snapshot_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO intake_context_snapshots
                (snapshot_id, conversation_id, created_at, context_json,
                 completeness_matching, completeness_referral, trigger_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    conversation_id,
                    now,
                    json.dumps(context.to_dict()),
                    context.completeness_for_matching(),
                    context.completeness_for_referral(),
                    trigger_message_id,
                ),
            )
            conn.commit()
        return snapshot_id

    def get_latest_context(self, conversation_id: str) -> IntakeContext:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT context_json FROM intake_context_snapshots
                WHERE conversation_id =?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        if not row:
            return IntakeContext()
        return IntakeContext.from_dict(json.loads(row["context_json"]))

    def get_transcript_excerpt(
        self,
        conversation_id: str,
        max_messages: int = 20,
    ) -> str:
        messages = self.get_messages(conversation_id)
        excerpt = messages[-max_messages:] if len(messages) > max_messages else messages
        lines = [f"{m.role.upper()}: {m.content}" for m in excerpt]
        return "\n\n".join(lines)
