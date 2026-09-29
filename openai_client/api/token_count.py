"""The official input-token counter (CONTEXT_METER_SPEC §3.1).

`client.responses.input_tokens.count` takes a SUBSET of what `responses.create` takes, keyword-
only and with no `**kwargs` — so a create body passed through unchanged is a TypeError, and one
that still carried `stream` or `store` would not describe any request the counter knows. The body
is therefore rebuilt from an allowlist, never forwarded.

The count is a METER, not a gate: it never raises for its own failure (only a cancellation, which
is the caller ending, propagates). A count that could not be taken answers `CountResult(None,
False)`, and every decision made from it treats "unknown" as "proceed, the API is the judge".
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

# The keyword parameters `AsyncInputTokens.count` accepts (SDK 3.19.x), and nothing else. Every
# create-only key — temperature, top_p, max_output_tokens, store, stream, include, the cache
# params, service_tier — and every wrapper-only kwarg falls away by not being listed.
COUNT_KEYS = ("conversation", "input", "instructions", "model", "parallel_tool_calls",
              "personality", "previous_response_id", "reasoning", "text", "tool_choice",
              "tools", "truncation")


@dataclass(frozen=True)
class CountResult:
    """One count. `tokens` is None when no count could be taken; `complete` is False whenever the
    number does not describe the whole request (a failure, or a count taken without MCP tools)."""

    tokens: Optional[int]
    complete: bool


def count_body(create_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """A NEW dict holding only the keys the counter accepts. Never mutates its argument.

    `input` is copied one level deep (a new list, same items): the tool loop appends to its running
    input between rounds, and a parallel count scheduled for round N must describe round N."""
    body = {key: create_kwargs[key] for key in COUNT_KEYS if key in create_kwargs}
    if isinstance(body.get("input"), list):
        body["input"] = list(body["input"])
    return body


def _without_mcp(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The body minus its MCP tools, or None when it carries none (nothing to retry without).

    The count reaches the MCP server for its tool list, so an unreachable server fails the count
    exactly as it would fail the request — and the rest of the request is still worth measuring."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return None
    kept = [t for t in tools if not (isinstance(t, dict) and t.get("type") == "mcp")]
    if len(kept) == len(tools):
        return None
    return {**body, "tools": kept}


async def count_input_tokens(client: Any, create_kwargs: Dict[str, Any]) -> CountResult:
    """Count one request's input tokens with OpenAI's own counter.

    `client` is the OpenAIClient (its `_safe_api_call` supplies the timeout and the WARNING log
    on failure). On failure, a body carrying MCP tools is counted ONCE more without them and the
    result is marked incomplete; any other failure answers `CountResult(None, False)`."""
    body = count_body(create_kwargs)
    try:
        response = await client._safe_api_call(
            client.client.responses.input_tokens.count,
            operation_type="token_count", **body)
        return CountResult(int(response.input_tokens), True)
    except asyncio.CancelledError:
        raise
    except Exception as first_error:  # noqa: BLE001 — a meter never fails a turn
        client.log_warning(f"Input-token count failed: {first_error}")
        retry = _without_mcp(body)
        if retry is None:
            return CountResult(None, False)
    try:
        response = await client._safe_api_call(
            client.client.responses.input_tokens.count,
            operation_type="token_count", **retry)
        return CountResult(int(response.input_tokens), False)
    except asyncio.CancelledError:
        raise
    except Exception as retry_error:  # noqa: BLE001
        client.log_warning(f"Input-token count without MCP tools failed: {retry_error}")
        return CountResult(None, False)


__all__ = ["COUNT_KEYS", "CountResult", "count_body", "count_input_tokens"]
