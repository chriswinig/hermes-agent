import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.session import SessionSource
from gateway.run import GatewayRunner


def _make_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner.hooks = MagicMock()
    runner.hooks.loaded_hooks = []
    runner._load_reasoning_config = MagicMock(return_value={})
    runner._resolve_turn_agent_config = MagicMock(return_value={
        "model": "gpt-5.4",
        "runtime": {"api_key": "test-key", "provider": "openai", "base_url": None, "api_mode": None},
    })
    return runner


def _make_source(user_id: str):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-5199641784",
        chat_type="group",
        user_id=user_id,
        user_name="tester",
    )


def test_load_gateway_config_parses_channel_restrictions(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
channel_restrictions:
  "-5199641784":
    owner_bypass: "982506524"
    brain_scope: "projects/lune/"
    channel_prompt: "Keep responses under 150 words. No preambles. No option lists longer than 3 items. Do not restate what the user just said. Match the energy of a sharp creative director in a Slack channel — punchy, direct, decisive."
    allowed_tools: [image_generate]
    blocked_tools: [terminal, search_files, read_file]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    cfg = load_gateway_config()

    assert cfg.channel_restrictions["-5199641784"]["owner_bypass"] == "982506524"
    assert cfg.channel_restrictions["-5199641784"]["brain_scope"] == "projects/lune/"
    assert "Keep responses under 150 words." in cfg.channel_restrictions["-5199641784"]["channel_prompt"]
    assert cfg.channel_restrictions["-5199641784"]["allowed_tools"] == ["image_generate"]
    assert cfg.channel_restrictions["-5199641784"]["blocked_tools"] == ["terminal", "search_files", "read_file"]


@pytest.mark.asyncio
async def test_run_agent_applies_channel_restrictions_for_non_owner():
    runner = _make_runner()
    source = _make_source("111")
    captured = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            captured["init"] = kwargs
            self.tools = []

        def run_conversation(self, *args, **kwargs):
            captured["run"] = {"args": args, "kwargs": kwargs}
            return {"final_response": "ok", "messages": [], "api_calls": 1, "completed": True}

    user_config = {
        "channel_restrictions": {
            "-5199641784": {
                "owner_bypass": "982506524",
                "brain_scope": "projects/lune/",
                "channel_prompt": "Keep responses under 150 words. No preambles. No option lists longer than 3 items. Do not restate what the user just said. Match the energy of a sharp creative director in a Slack channel — punchy, direct, decisive.",
                "allowed_tools": ["image_generate"],
                "blocked_tools": ["terminal", "search_files", "read_file"],
            }
        }
    }

    with patch("run_agent.AIAgent", FakeAgent), \
         patch("gateway.run._load_gateway_config", return_value=user_config), \
         patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), \
         patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***", "provider": "openai", "base_url": None, "api_mode": None}), \
         patch("hermes_cli.tools_config._get_platform_tools", return_value={"terminal", "file", "web", "image_gen"}), \
         patch("gateway.run.load_dotenv"):
        result = await runner._run_agent(
            message="hello",
            context_prompt="base context",
            history=[],
            source=source,
            session_id="sid",
            session_key="skey",
        )

    assert result["final_response"] == "ok"
    assert captured["init"].get("allowed_tools") == ["image_generate"]
    assert captured["init"].get("disabled_tools") == ["read_file", "search_files", "terminal"]
    assert "Keep responses under 150 words." in captured["init"]["ephemeral_system_prompt"]
    assert "projects/lune/" in captured["init"]["ephemeral_system_prompt"]
    assert "image_generate" in captured["init"]["ephemeral_system_prompt"]
    assert "982506524" not in captured["init"]["ephemeral_system_prompt"]


@pytest.mark.asyncio
async def test_run_agent_owner_bypass_skips_channel_restrictions():
    runner = _make_runner()
    source = _make_source("982506524")
    captured = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            captured["init"] = kwargs
            self.tools = []

        def run_conversation(self, *args, **kwargs):
            return {"final_response": "ok", "messages": [], "api_calls": 1, "completed": True}

    user_config = {
        "channel_restrictions": {
            "-5199641784": {
                "owner_bypass": "982506524",
                "brain_scope": "projects/lune/",
                "channel_prompt": "Keep responses under 150 words. No preambles. No option lists longer than 3 items. Do not restate what the user just said. Match the energy of a sharp creative director in a Slack channel — punchy, direct, decisive.",
                "allowed_tools": ["image_generate"],
                "blocked_tools": ["terminal", "search_files", "read_file"],
            }
        }
    }

    with patch("run_agent.AIAgent", FakeAgent), \
         patch("gateway.run._load_gateway_config", return_value=user_config), \
         patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), \
         patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key", "provider": "openai", "base_url": None, "api_mode": None}), \
         patch("hermes_cli.tools_config._get_platform_tools", return_value={"terminal", "file", "web"}), \
         patch("gateway.run.load_dotenv"):
        await runner._run_agent(
            message="hello",
            context_prompt="base context",
            history=[],
            source=source,
            session_id="sid",
            session_key="skey",
        )

    assert captured["init"].get("allowed_tools") in (None, [])
    assert captured["init"].get("disabled_tools") in (None, [])
    assert captured["init"]["ephemeral_system_prompt"].startswith("base context")
    assert "Keep responses under 150 words." in captured["init"]["ephemeral_system_prompt"]
