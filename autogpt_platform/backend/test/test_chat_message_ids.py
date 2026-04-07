import uuid

import pytest
import pytest_asyncio

from backend.copilot.db import (
    add_chat_messages_batch,
    create_chat_session as create_chat_session_record,
    get_chat_session as get_persisted_chat_session,
)
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    upsert_chat_session,
)
from backend.data.user import get_or_create_user


@pytest.fixture
def test_user_id() -> str:
    return str(uuid.uuid4())


@pytest_asyncio.fixture
async def setup_test_user(test_user_id):
    user_data = {
        "sub": test_user_id,
        "email": f"{test_user_id}@example.com",
        "user_metadata": {"name": "Test User"},
    }
    await get_or_create_user(user_data)
    return test_user_id


@pytest.mark.asyncio
async def test_batch_insert_preserves_explicit_message_id_and_duration(
    setup_test_user,
    test_user_id,
):
    session_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    duration_ms = 321

    await create_chat_session_record(session_id=session_id, user_id=test_user_id)

    next_sequence = await add_chat_messages_batch(
        session_id=session_id,
        messages=[
            {
                "id": message_id,
                "role": "assistant",
                "content": "hello from assistant",
                "duration_ms": duration_ms,
            }
        ],
        start_sequence=0,
    )

    assert next_sequence == 1

    persisted_session = await get_persisted_chat_session(session_id)
    assert persisted_session is not None
    assert len(persisted_session.messages) == 1
    assert persisted_session.messages[0].id == message_id
    assert persisted_session.messages[0].duration_ms == duration_ms


@pytest.mark.asyncio
async def test_chat_session_save_preserves_explicit_message_id_and_duration(
    setup_test_user,
    test_user_id,
):
    message_id = str(uuid.uuid4())
    duration_ms = 321

    session = ChatSession.new(user_id=test_user_id, dry_run=False)
    session.messages = [
        ChatMessage(
            id=message_id,
            role="assistant",
            content="hello from assistant",
            duration_ms=duration_ms,
        )
    ]

    await upsert_chat_session(session)

    loaded_session = await get_chat_session(session.session_id, test_user_id)
    assert loaded_session is not None
    assert len(loaded_session.messages) == 1
    assert loaded_session.messages[0].role == "assistant"
    assert loaded_session.messages[0].content == "hello from assistant"
    assert loaded_session.messages[0].id == message_id
    assert loaded_session.messages[0].duration_ms == duration_ms
