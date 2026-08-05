"""Supply Chain — scan tool docstrings before registering them on an agent
(PROJECT_PLAN.md SS7, category 3; reference/notebook_patterns.md SS2c).

As of 2026-08-05 this is no longer hypothetical: `backend/agents/policy_rag.py`
and `backend/agents/fraud_detection.py` bind their tools via
`backend.mcp_server.client_tools` (an MCP-transport wrapper), not a direct
`@tool`-decorated import — this is exactly the "external, less-trusted
source" scenario this module was originally built for.

**Real bug found and fixed while wiring that up:** the original
`_tool_docstring` only checked `tool.func.__doc__` — correct for a
`@tool`-decorated function (where LangChain derives `.description` from
the docstring automatically), but `StructuredTool.from_function(func=X,
description=Y)` — what the MCP client adapter uses, since the wrapped
function has no docstring of its own — sets `.description` directly,
leaving `.func.__doc__` empty. The scanner was silently checking an empty
string for every MCP-transport tool, providing zero real protection.
Fixed by checking `tool.description` (what's actually sent to Claude)
*and* the unwrapped function's docstring, covering both construction
styles."""

import re
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool

# Illustrative denylist, not exhaustive — extend as new attack phrasing
# shows up in practice, same spirit as the injection-pattern list.
_SUSPICIOUS_DOCSTRING_PATTERN = re.compile(
    r"SYSTEM OVERRIDE|ignore.*instructions|password|reveal.*prompt", re.IGNORECASE
)


def _tool_docstring(tool: BaseTool | Callable[..., Any]) -> str:
    # `@tool`-decorated functions wrap the original callable in a
    # StructuredTool — `.func` is the real, unwrapped function whose
    # docstring an attacker actually controls. Checking the StructuredTool
    # wrapper's own (LangChain-authored) docstring would miss this
    # entirely, hence the explicit .func unwrap. `.description` is checked
    # too — the field actually sent to the LLM as the tool's description,
    # which for `StructuredTool.from_function(description=...)`-built
    # tools (e.g. the MCP client adapter) is set directly and never flows
    # through any function's `__doc__` at all.
    unwrapped = getattr(tool, "func", tool)
    func_doc = getattr(unwrapped, "__doc__", None) or ""
    description = getattr(tool, "description", None) or ""
    return f"{func_doc}\n{description}"


def is_tool_safe(tool: BaseTool | Callable[..., Any]) -> bool:
    return _SUSPICIOUS_DOCSTRING_PATTERN.search(_tool_docstring(tool)) is None


def validate_registered_tools(tools: list[BaseTool | Callable[..., Any]]) -> None:
    """Raise if any tool's docstring trips the denylist. Called once at
    agent-construction time (backend/agents/policy_rag.py,
    backend/agents/fraud_detection.py) — cheap (a regex over a short
    string), so there's no reason to skip it even for today's fully-trusted
    local tools."""
    unsafe = [getattr(t, "name", getattr(t, "__name__", repr(t))) for t in tools if not is_tool_safe(t)]
    if unsafe:
        raise ValueError(f"Refusing to register tool(s) with suspicious docstrings: {unsafe}")
