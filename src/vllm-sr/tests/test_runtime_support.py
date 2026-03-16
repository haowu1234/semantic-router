from cli.commands.runtime_support import (
    apply_runtime_mode_env_vars,
    append_passthrough_env_vars,
)


def test_apply_runtime_mode_env_vars_sets_dashboard_readonly_when_requested():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=False,
        readonly=True,
        setup_mode=False,
        platform=None,
    )

    assert env_vars["DASHBOARD_READONLY"] == "true"


def test_apply_runtime_mode_env_vars_skips_dashboard_readonly_in_minimal_mode():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=True,
        readonly=True,
        setup_mode=False,
        platform=None,
    )

    assert env_vars["DISABLE_DASHBOARD"] == "true"
    assert "DASHBOARD_READONLY" not in env_vars


def test_append_passthrough_env_vars_forwards_nl_planner_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://host.docker.internal:8002/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("NL_PLANNER_BACKEND", "tool-calling-llm")
    monkeypatch.setenv("NL_PLANNER_PROVIDER", "openai-compatible")
    monkeypatch.setenv("NL_PLANNER_BASE_URL", "http://vllm-gpt-oss-120b:8000/v1")
    monkeypatch.setenv("NL_PLANNER_API_KEY", "planner-secret")
    monkeypatch.setenv("NL_PLANNER_MODEL", "DeepSeek-V3.2")
    monkeypatch.setenv("NL_PLANNER_TIMEOUT_MS", "30000")
    monkeypatch.setenv("NL_PLANNER_MAX_OUTPUT_TOKENS", "1800")
    monkeypatch.setenv("NL_PLANNER_TOOL_BUDGET", "4")
    monkeypatch.setenv("NL_PLANNER_ALLOW_WEB_TOOLS", "false")
    monkeypatch.setenv("NL_PLANNER_ALLOW_MCP_TOOLS", "false")
    monkeypatch.setenv("NL_PLANNER_ALLOWED_MCP_TOOLS", "fetch_raw_url")

    env_vars: dict[str, str] = {}

    append_passthrough_env_vars(env_vars)

    assert env_vars["OPENAI_BASE_URL"] == "http://host.docker.internal:8002/v1"
    assert env_vars["OPENAI_API_KEY"] == "openai-secret"
    assert env_vars["NL_PLANNER_BACKEND"] == "tool-calling-llm"
    assert env_vars["NL_PLANNER_PROVIDER"] == "openai-compatible"
    assert env_vars["NL_PLANNER_BASE_URL"] == "http://vllm-gpt-oss-120b:8000/v1"
    assert env_vars["NL_PLANNER_API_KEY"] == "planner-secret"
    assert env_vars["NL_PLANNER_MODEL"] == "DeepSeek-V3.2"
    assert env_vars["NL_PLANNER_TIMEOUT_MS"] == "30000"
    assert env_vars["NL_PLANNER_MAX_OUTPUT_TOKENS"] == "1800"
    assert env_vars["NL_PLANNER_TOOL_BUDGET"] == "4"
    assert env_vars["NL_PLANNER_ALLOW_WEB_TOOLS"] == "false"
    assert env_vars["NL_PLANNER_ALLOW_MCP_TOOLS"] == "false"
    assert env_vars["NL_PLANNER_ALLOWED_MCP_TOOLS"] == "fetch_raw_url"
