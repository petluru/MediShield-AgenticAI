import json
import uuid

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from backend.agents.llm_factory import (
    _TokenUsageLogger,
    build_chat_anthropic,
    cached_system_message,
    estimate_cost,
)
from backend.config import Settings


def make_settings(**overrides):
    defaults = {
        "_env_file": None,
        "ANTHROPIC_API_KEY": "sk-ant-test",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def make_result(
    response_id: str,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> LLMResult:
    usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_creation or cache_read:
        usage_metadata["input_token_details"] = {"cache_creation": cache_creation, "cache_read": cache_read}
    message = AIMessage(
        content="ok",
        usage_metadata=usage_metadata,
        response_metadata={"id": response_id},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_build_chat_anthropic_uses_configured_model_by_default():
    llm = build_chat_anthropic("claude-sonnet-5", make_settings(), agent="test")
    assert llm.model == "claude-sonnet-5"


def test_build_chat_anthropic_uses_testing_model_when_flag_set():
    settings = make_settings(USE_CHEAP_MODELS_FOR_TESTING=True, TESTING_MODEL="claude-haiku-4-5")
    llm = build_chat_anthropic("claude-sonnet-5", settings, agent="test")
    assert llm.model == "claude-haiku-4-5"


def test_cached_system_message_wraps_prompt_with_cache_control():
    message = cached_system_message("be terse")
    assert isinstance(message, SystemMessage)
    assert message.content == [{"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}]


def test_token_usage_logger_writes_a_row_for_a_fresh_response(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    logger = _TokenUsageLogger(log_path, agent="classifier", model_name="claude-sonnet-5")

    logger.on_llm_end(make_result("msg_1"), run_id=None)  # type: ignore[arg-type]

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["response_id"] == "msg_1"
    assert rows[0]["input_tokens"] == 100
    assert rows[0]["agent"] == "classifier"


def test_token_usage_logger_dedupes_cache_hits_by_response_id(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    logger = _TokenUsageLogger(log_path, agent="classifier", model_name="claude-sonnet-5")

    fresh = make_result("msg_dup")
    logger.on_llm_end(fresh, run_id=None)  # type: ignore[arg-type]
    logger.on_llm_end(fresh, run_id=None)  # type: ignore[arg-type]  # SQLiteCache replay: same id
    logger.on_llm_end(fresh, run_id=None)  # type: ignore[arg-type]

    rows = log_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


def test_token_usage_logger_logs_distinct_responses_separately(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    logger = _TokenUsageLogger(log_path, agent="classifier", model_name="claude-sonnet-5")

    logger.on_llm_end(make_result("msg_a"), run_id=None)  # type: ignore[arg-type]
    logger.on_llm_end(make_result("msg_b"), run_id=None)  # type: ignore[arg-type]

    rows = log_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2


def test_token_usage_logger_captures_cache_tokens_and_cost(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    logger = _TokenUsageLogger(log_path, agent="policy_rag", model_name="claude-sonnet-5")

    logger.on_llm_end(
        make_result("msg_cached", input_tokens=1322, output_tokens=124, cache_creation=1322, cache_read=8871),
        run_id=None,  # type: ignore[arg-type]
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["cache_creation_input_tokens"] == 1322
    assert row["cache_read_input_tokens"] == 8871
    assert row["estimated_cost"] == round(estimate_cost("claude-sonnet-5", 1322, 124, 1322, 8871), 6)


def test_token_usage_logger_defaults_cache_fields_to_zero_when_absent(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    logger = _TokenUsageLogger(log_path, agent="classifier", model_name="claude-sonnet-5")

    logger.on_llm_end(make_result("msg_no_cache"), run_id=None)  # type: ignore[arg-type]

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["cache_creation_input_tokens"] == 0
    assert row["cache_read_input_tokens"] == 0


def test_token_usage_logger_measures_latency_between_start_and_end(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    logger = _TokenUsageLogger(log_path, agent="classifier", model_name="claude-sonnet-5")
    run_id = uuid.uuid4()

    logger.on_chat_model_start({}, [[]], run_id=run_id)
    logger.on_llm_end(make_result("msg_timed"), run_id=run_id)

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["latency_ms"] is not None
    assert row["latency_ms"] >= 0


def test_token_usage_logger_latency_is_none_without_a_matching_start(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    logger = _TokenUsageLogger(log_path, agent="classifier", model_name="claude-sonnet-5")

    logger.on_llm_end(make_result("msg_no_start"), run_id=uuid.uuid4())

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["latency_ms"] is None


def test_estimate_cost_accounts_for_cache_write_and_read_multipliers():
    # 1M regular input tokens at $2/M = $2.00; 1M cache-write tokens at
    # 1.25x = $2.50; 1M cache-read tokens at 0.1x = $0.20; no output tokens.
    cost = estimate_cost("claude-sonnet-5", 1_000_000, 0, 1_000_000, 1_000_000)
    assert round(cost, 2) == round(2.00 + 2.50 + 0.20, 2)
