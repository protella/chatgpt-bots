from __future__ import annotations

from typing import Any, Dict, Optional

from config import config


class SlackAssistantEventsMixin:
    """Agent split-view (Assistant surface) adapter.

    Handles the assistant-thread lifecycle events additively: greeting + suggested prompts
    on thread start, and best-effort thread titles. User messages in assistant threads are
    ordinary ``message.im`` events and keep flowing through the existing DM path — this
    mixin never touches the message pipeline. Everything here is best-effort/never-raise,
    mirroring set_assistant_status.

    NOTE: slack_bolt's AsyncAssistant middleware was deliberately NOT used — it registers
    its own listener for message.im events inside assistant threads
    (is_user_message_event_in_assistant_thread) and would absorb them before our
    existing @app.event("message") flow.
    """

    @staticmethod
    def _extract_assistant_thread(event: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Pull (channel_id, thread_ts) out of an assistant_thread_* event, defensively."""
        thread = event.get("assistant_thread")
        if not isinstance(thread, dict):
            return None
        channel_id = thread.get("channel_id")
        thread_ts = thread.get("thread_ts")
        if not channel_id or not thread_ts:
            return None
        return {"channel_id": channel_id, "thread_ts": thread_ts}

    async def _handle_assistant_thread_started(self, event: Dict[str, Any], client) -> None:
        """User opened the split view: greet them and set the suggested starter prompts."""
        if not config.enable_assistant_surface:
            return
        thread = self._extract_assistant_thread(event)
        if not thread:
            self.log_debug("assistant_thread_started without a usable assistant_thread payload; skipping")
            return

        # Greeting (a normal threaded message in the assistant container)
        try:
            if config.assistant_greeting:
                await self.app.client.chat_postMessage(
                    channel=thread["channel_id"],
                    thread_ts=thread["thread_ts"],
                    text=config.assistant_greeting,
                )
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"Could not post assistant greeting: {e}")

        # Suggested prompts (title max length is enforced by Slack; keep titles short)
        try:
            prompts = [
                {"title": (p[:37] + "…") if len(p) > 38 else p, "message": p}
                for p in (config.assistant_suggested_prompts or [])
                if p and p.strip()
            ][:4]
            if prompts and hasattr(self.app.client, "assistant_threads_setSuggestedPrompts"):
                await self.app.client.assistant_threads_setSuggestedPrompts(
                    channel_id=thread["channel_id"],
                    thread_ts=thread["thread_ts"],
                    prompts=prompts,
                )
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"Could not set suggested prompts: {e}")

    async def _handle_assistant_thread_context_changed(self, event: Dict[str, Any]) -> None:
        """User switched channels while the split view is open. Context use is a future item —
        this handler exists so the subscribed event isn't unhandled noise."""
        thread = self._extract_assistant_thread(event) or {}
        self.log_debug(
            f"assistant_thread_context_changed: channel={thread.get('channel_id')}, "
            f"context={ (event.get('assistant_thread') or {}).get('context') }"
        )

    async def _maybe_set_assistant_thread_title(self, channel_id: str, thread_ts: str, text: str) -> None:
        """Best-effort assistant.threads.setTitle from the first user message (once per thread).

        Only meaningful for assistant threads (DM channels); harmlessly no-ops elsewhere —
        Slack rejects the call for non-assistant threads and we just log at debug.
        """
        if not config.enable_assistant_surface:
            return
        if not channel_id or not channel_id.startswith("D") or not thread_ts:
            return
        title_source = (text or "").strip()
        if not title_source:
            return
        titled = getattr(self, "_titled_assistant_threads", None)
        if titled is None:
            titled = set()
            self._titled_assistant_threads = titled
        key = f"{channel_id}:{thread_ts}"
        if key in titled:
            return
        titled.add(key)  # mark first (even on failure) so we never retry-spam setTitle
        title = title_source if len(title_source) <= 60 else title_source[:59] + "…"
        try:
            if hasattr(self.app.client, "assistant_threads_setTitle"):
                await self.app.client.assistant_threads_setTitle(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    title=title,
                )
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"assistant setTitle unavailable here ({e}); continuing without it")
