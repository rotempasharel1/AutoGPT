import pytest

from backend.copilot.db import (
    add_chat_messages_batch,
    create_chat_session,
    get_chat_session,
)

@pytest.mark.asyncio
async def test_batch_insert_preserves_explicit_message_id():
    """Verify batch insertion preserves an explicit message ID."""

@pytest.mark.asyncio
async def test_batch_insert_preserves_explicit_message_id():
    session_id = "test-session-id-stable-message"
    user_id = "test-user-id"
    message_id = "test-message-id-123"

    await create_chat_session(session_id=session_id, user_id=user_id)

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

    session = await get_chat_session(session_id)
    assert session is not None
    assert len(session.messages) == 1
    assert session.messages[0].role == "assistant"
    assert session.messages[0].content == "hello from assistant"
    assert session.messages[0].id == message_id
