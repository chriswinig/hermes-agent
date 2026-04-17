import asyncio
import json
import threading
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import hermes_cli.runtime_provider as runtime_provider

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import (
    GatewayRunner,
    _resolve_entity_detection_runtime_kwargs,
)
from gateway.session import SessionEntry, SessionSource


class _StubAdapter:
    async def send(self, *args, **kwargs):
        return None


def _make_event(text="Acme met John", platform=Platform.TELEGRAM):
    source = SessionSource(
        platform=platform,
        user_id="user-1",
        chat_id="chat-1",
        user_name="Chris",
        chat_type="dm",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )


def _make_runner(tmp_path):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner.adapters = {Platform.TELEGRAM: _StubAdapter()}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._smart_model_routing = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._background_tasks = set()
    runner._entity_detection_lock = threading.Lock()
    runner._show_reasoning = False
    runner._update_prompt_pending = {}
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.delivery_router = MagicMock()
    runner._set_session_env = lambda _context: None
    runner._clear_session_env = lambda _tokens=None: None
    runner._has_setup_skill = lambda: False
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    return runner


@pytest.mark.asyncio
async def test_schedule_entity_detection_spawns_background_task(tmp_path):
    runner = _make_runner(tmp_path)
    event = _make_event()
    created_tasks = []

    def fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        created_tasks.append(task)
        return task

    with patch("gateway.run.asyncio.create_task", side_effect=fake_create_task):
        runner._schedule_entity_detection(
            event=event,
            message_text="Acme met John in Boston.",
            session_key="session-key",
            session_id="session-1",
        )

    assert len(created_tasks) == 1
    assert created_tasks[0] in runner._background_tasks


@pytest.mark.asyncio
async def test_run_entity_detection_task_uses_gpt_4_1_mini_and_persists_jsonl(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    event = _make_event()
    captured = {}

    import run_agent

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def run_conversation(self, user_message, task_id=None, **kwargs):
            captured["user_message"] = user_message
            captured["task_id"] = task_id
            return {
                "final_response": json.dumps(
                    {
                        "entities": [
                            {
                                "name": "Acme",
                                "type": "company",
                                "confidence": 0.93,
                                "evidence": "Acme",
                            },
                            {
                                "name": "John",
                                "type": "person",
                                "confidence": 0.81,
                                "evidence": "John",
                            },
                        ]
                    }
                )
            }

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_entity_detection_runtime_kwargs",
        lambda: {
            "api_key": "***",
            "base_url": "https://example.invalid/v1",
            "provider": "custom",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
            "credential_pool": None,
            "source": "explicit",
        },
    )

    recorded = {}
    real_wait_for = asyncio.wait_for

    async def capture_wait(awaitable, timeout):
        recorded["timeout"] = timeout
        return await real_wait_for(awaitable, timeout=timeout)

    with patch("gateway.run.asyncio.wait_for", side_effect=capture_wait):
        await runner._run_entity_detection_task(
            event=event,
            message_text="Acme met John in Boston.",
            session_key="session-key",
            session_id="session-1",
        )

    record_path = tmp_path / "sessions" / "entity_detections" / "session-1.jsonl"
    assert record_path.exists()
    rows = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["model"] == "gpt-4.1-mini"
    assert row["message_id"] == "msg-1"
    assert [entity["name"] for entity in row["entities"]] == ["Acme", "John"]

    assert captured["init"]["model"] == "gpt-4.1-mini"
    assert captured["init"]["enabled_toolsets"] == []
    assert captured["init"]["reasoning_config"] == {"enabled": False}
    assert captured["init"]["skip_memory"] is True
    assert captured["init"]["skip_context_files"] is True
    assert captured["init"]["persist_session"] is False
    assert recorded["timeout"] == 120
    assert "Return strict JSON" in captured["user_message"]


@pytest.mark.asyncio
async def test_run_entity_detection_task_records_model_errors(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    event = _make_event()

    import run_agent

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_conversation(self, user_message, task_id=None, **kwargs):
            return {
                "final_response": None,
                "failed": True,
                "error": "model unsupported on this provider",
            }

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_entity_detection_runtime_kwargs",
        lambda: {
            "api_key": "***",
            "base_url": "https://example.invalid/v1",
            "provider": "custom",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
            "credential_pool": None,
            "source": "explicit",
        },
    )

    await runner._run_entity_detection_task(
        event=event,
        message_text="Meeting with Marlene at wingate tomorrow",
        session_key="session-key",
        session_id="session-1",
    )

    record_path = tmp_path / "sessions" / "entity_detections" / "session-1.jsonl"
    rows = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "error"
    assert row["error"] == "model unsupported on this provider"
    assert row["entities"] == []


def test_resolve_entity_detection_runtime_kwargs_uses_explicit_override(monkeypatch):
    monkeypatch.setenv("HERMES_ENTITY_DETECTION_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("HERMES_ENTITY_DETECTION_API_KEY", "sk-test")
    monkeypatch.delenv("HERMES_ENTITY_DETECTION_PROVIDER", raising=False)

    captured = {}

    def fake_resolve_runtime_provider(*, requested=None, explicit_api_key=None, explicit_base_url=None):
        captured["requested"] = requested
        captured["explicit_api_key"] = explicit_api_key
        captured["explicit_base_url"] = explicit_base_url
        return {
            "provider": "custom",
            "api_mode": "codex_responses",
            "base_url": explicit_base_url,
            "api_key": explicit_api_key,
            "source": "explicit",
            "credential_pool": None,
            "command": None,
            "args": [],
        }

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", fake_resolve_runtime_provider)

    runtime = _resolve_entity_detection_runtime_kwargs()

    assert captured["requested"] == "custom"
    assert captured["explicit_base_url"] == "https://api.openai.com/v1"
    assert captured["explicit_api_key"] == "sk-test"
    assert runtime["provider"] == "custom"
    assert runtime["api_mode"] == "codex_responses"


@pytest.mark.asyncio
async def test_run_entity_detection_task_skips_incompatible_codex_runtime(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    event = _make_event(text="Meeting with Marlene at wingate tomorrow")

    monkeypatch.setattr(
        "gateway.run._resolve_entity_detection_runtime_kwargs",
        lambda: {
            "api_key": "***",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
            "credential_pool": None,
            "source": "device_code",
        },
    )

    await runner._run_entity_detection_task(
        event=event,
        message_text="Meeting with Marlene at wingate tomorrow",
        session_key="session-key",
        session_id="session-1",
    )

    record_path = tmp_path / "sessions" / "entity_detections" / "session-1.jsonl"
    rows = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "skipped"
    assert "gpt-4.1-mini is not supported" in row["error"]
    assert row["runtime_provider"] == "openai-codex"
    assert row["runtime_source"] == "device_code"


@pytest.mark.asyncio
async def test_handle_message_with_agent_starts_entity_detection_before_main_agent(tmp_path):
    runner = _make_runner(tmp_path)
    event = _make_event(text="Acme met John")
    session_entry = SessionEntry(
        session_key="session-key",
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=event.source,
        display_name="Test",
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    runner._schedule_entity_detection = MagicMock()
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "hello back",
            "messages": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    result = await runner._handle_message_with_agent(event, event.source, "session-key")

    assert result == "hello back"
    runner._schedule_entity_detection.assert_called_once()
    call = runner._schedule_entity_detection.call_args.kwargs
    assert call["session_key"] == "session-key"
    assert call["session_id"] == "session-1"
    assert call["message_text"] == "Acme met John"
