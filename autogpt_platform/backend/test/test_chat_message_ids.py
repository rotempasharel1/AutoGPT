from uuid import uuid4

import pytest

from backend.copilot.db import (
    add_chat_messages_batch,
    create_chat_session,
    get_chat_session as get_db_chat_session,
)
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    upsert_chat_session,
)
from backend.data.redis_client import get_redis_async

@pytest.mark.asyncio
async def test_batch_insert_preserves_explicit_message_id():
    """Verify batch insertion preserves an explicit message ID."""

@pytest.mark.asyncio(loop_scope="session")
async def test_batch_insert_preserves_explicit_message_id(
    setup_test_user, test_user_id
):
    session_id = str(uuid4())
    message_id = str(uuid4())

    await create_chat_session(session_id=session_id, user_id=test_user_id)

    next_sequence = await add_chat_messages_batch(
        session_id=session_id,
        messages=[
            {
                "id": message_id,
                "role": "assistant",
                "content": "hello from assistant",
            }
        ],
        start_sequence=0,
    )

    assert next_sequence == 1

    session = await get_db_chat_session(session_id)
    assert session is not None
    assert len(session.messages) == 1
    assert session.messages[0].role == "assistant"
    assert session.messages[0].content == "hello from assistant"
    assert session.messages[0].id == message_id


@pytest.mark.asyncio(loop_scope="session")
async def test_upsert_chat_session_persists_message_ids_and_duration(
    setup_test_user, test_user_id
):
    session = ChatSession.new(user_id=test_user_id, dry_run=False)
    user_message_id = str(uuid4())
    assistant_message_id = str(uuid4())
    session.messages = [
        ChatMessage(id=user_message_id, role="user", content="hello"),
        ChatMessage(
            id=assistant_message_id,
            role="assistant",
            content="hi there",
            duration_ms=1234,
        ),
    ]

    session = await upsert_chat_session(session)

    async_redis = await get_redis_async()
    await async_redis.delete(f"chat:session:{session.session_id}")

    loaded = await get_chat_session(session.session_id, test_user_id)
    assert loaded is not None
    assert [msg.id for msg in loaded.messages] == [
        user_message_id,
        assistant_message_id,
    ]
    assert loaded.messages[1].duration_ms == 1234
