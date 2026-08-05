"""Shared ChatAnthropic construction for every agent — keeps model names
driven entirely by Settings (PROJECT_PLAN.md SS4: never hardcode them), and
centralizes the cost-control mechanisms added after the credit-usage
discussion: exact-match response caching, a cheap-model override for smoke
testing, per-call token usage logging, and an Anthropic prompt-cache helper
for system prompts."""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from langchain_anthropic import ChatAnthropic
from langchain_community.cache import SQLiteCache
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.globals import set_llm_cache
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import LLMResult
from pydantic import SecretStr

from backend.config import Settings

_cache_initialized = False
_cache_lock = threading.Lock()

# $ per million tokens (regular input, output). The single source of truth —
# `backend/scripts/token_usage_report.py` imports this rather than
# duplicating it. Sonnet 5's introductory rate applies through 2026-08-31 —
# update after that date. Unrecognized models price at $0.
PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
}

# Anthropic prompt-caching multipliers on the base input rate (5-minute TTL,
# what AnthropicPromptCachingMiddleware defaults to): writing to the cache
# costs 1.25x a regular input token; reading a cache hit costs 0.1x.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.1


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    """Real per-call cost, accounting for Anthropic's cache write/read
    multipliers. `input_tokens` from `usage_metadata` already excludes
    `cache_creation`/`cache_read` tokens — Anthropic reports them as
    separate buckets, not overlapping subsets — so this sums all three
    without double-counting."""
    input_rate, output_rate = PRICING.get(model_name, (0.0, 0.0))
    return (
        input_tokens * input_rate
        + cache_creation_input_tokens * input_rate * _CACHE_WRITE_MULTIPLIER
        + cache_read_input_tokens * input_rate * _CACHE_READ_MULTIPLIER
        + output_tokens * output_rate
    ) / 1_000_000


def _ensure_llm_cache(settings: Settings) -> None:
    """Exact-match response cache (langchain SQLiteCache): a request whose
    prompt+image bytes exactly match a prior call is served from disk for
    free, no API call at all. Set once per process — `set_llm_cache` is
    process-global. Protects against wasted reruns when only non-prompt
    (e.g. validation) code changed since the last smoke test."""
    global _cache_initialized
    if _cache_initialized:
        return
    with _cache_lock:
        if _cache_initialized:
            return
        cache_path = settings.resolved_path(settings.llm_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(cache_path)))
        _cache_initialized = True


class _TokenUsageLogger(BaseCallbackHandler):
    """Appends one JSON line per completed LLM call to the token usage log —
    ground truth from `usage_metadata`, not an after-the-fact estimate.

    A SQLiteCache replay fires this same callback with a byte-identical
    response (verified empirically: same Anthropic message `id`, same
    `usage_metadata`, zero tokens actually spent) — logging it as-is would
    silently double-count every cache hit. We dedupe against response ids
    already present in the log file itself, not an in-memory set, since
    separate script invocations (the normal way smoke tests get rerun) are
    separate processes and still need to share the persistent SQLite cache's
    hit/miss history."""

    def __init__(self, log_path: Path, agent: str, model_name: str) -> None:
        self._log_path = log_path
        self._agent = agent
        self._model_name = model_name
        # Per-run_id start times for latency measurement. A `create_agent`
        # tool loop fires multiple overlapping runs (parallel tool calls run
        # on a thread pool, see backend/rag/ingest.py's client-singleton
        # comment for the same underlying concurrency), so this must be
        # keyed by run_id and lock-guarded, not a single shared timestamp.
        self._start_times: dict[UUID, float] = {}
        self._start_times_lock = threading.Lock()

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: object,
    ) -> None:
        with self._start_times_lock:
            self._start_times[run_id] = time.monotonic()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: object) -> None:
        with self._start_times_lock:
            start = self._start_times.pop(run_id, None)
        latency_ms = round((time.monotonic() - start) * 1000) if start is not None else None

        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                usage = getattr(message, "usage_metadata", None)
                if not usage:
                    continue
                response_metadata = getattr(message, "response_metadata", None) or {}
                response_id = response_metadata.get("id")
                if response_id is not None and self._already_logged(response_id):
                    continue
                self._append(usage, response_id, latency_ms)

    def _already_logged(self, response_id: str) -> bool:
        if not self._log_path.exists():
            return False
        with self._log_path.open(encoding="utf-8") as f:
            return any(json.loads(line).get("response_id") == response_id for line in f if line.strip())

    def _append(self, usage: dict, response_id: str | None, latency_ms: int | None) -> None:
        input_tokens = usage.get("input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        input_token_details = usage.get("input_token_details") or {}
        cache_creation = input_token_details.get("cache_creation") or 0
        cache_read = input_token_details.get("cache_read") or 0

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": self._agent,
            "model": self._model_name,
            "response_id": response_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens"),
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "estimated_cost": round(
                estimate_cost(self._model_name, input_tokens, output_tokens, cache_creation, cache_read), 6
            ),
            "latency_ms": latency_ms,
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def build_chat_anthropic(model_name: str, settings: Settings, agent: str = "unknown") -> ChatAnthropic:
    _ensure_llm_cache(settings)
    # Dev cost control: force every agent onto the cheap testing model when
    # the flag is set, regardless of which production model was requested.
    effective_model = settings.testing_model if settings.use_cheap_models_for_testing else model_name
    log_path = settings.resolved_path(settings.token_usage_log_path)

    # `temperature` is rejected outright ("deprecated for this model") by the
    # claude-5 family — omit it rather than pinning 0.
    return ChatAnthropic(
        model_name=effective_model,
        api_key=SecretStr(settings.anthropic_api_key),
        timeout=60,
        stop=None,
        callbacks=[_TokenUsageLogger(log_path, agent, effective_model)],
    )


def cached_system_message(prompt: str) -> SystemMessage:
    """Wrap a system prompt with an Anthropic prompt-cache breakpoint
    (`cache_control: ephemeral`). This is a no-op today — our prompts are
    under Sonnet 5's 1024-token cache minimum — but it's free to have on and
    activates automatically once prompts grow (e.g. task #12's security
    boilerplate). See IMPLEMENTATION_CHALLENGES.md."""
    return SystemMessage(content=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}])
