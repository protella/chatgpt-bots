---
name: impl-round
description: The standing workflow for any substantial change in this repo — codex alignment on the plan, sub-agents for the build, codex review of the diff, then capped tests and a live dev-bot pass. Use before starting any medium-or-larger implementation.
---

# Implementation round

## 1. Align the plan with codex

Scope a concrete plan, then run it past codex for an adversarial **alignment** review before any
code is written. Fix the gaps it finds first.

Use the `codex exec` CLI (the MCP is disabled), run it in the background, and capture the session
id so the review pass can resume the same thread. It needs `</dev/null` or it hangs.

## 2. Build with sub-agents

Dispatch sub-agents over **disjoint files** and coordinate with SendMessage. This keeps the
token-heavy work out of the main conversation. Don't pass an explicit `mode` — it stomps the
user's session permission mode.

## 3. Review the diff

Codex reviews the completed implementation adversarially, with `file:line` findings, and I review
alongside it. Ask reviewers to report **everything** and filter afterward; telling a reviewer to be
conservative or to report only high-severity issues makes it report less.

Don't re-run the suite or the linter a sub-agent already ran and reported with concrete numbers.
Verify cheaply instead: the commit exists, the tree is clean, the diff matches the design. Re-run
only when the report contradicts the diff or the agent died mid-run.

## 4. Test, then live-verify

Capped pytest (`run-tests` skill), then a real dev-bot pass (`live-bot-test` skill) before the work
counts as done.

## Throughout: UX trade-offs are the user's call

Review-driven fixes optimize for correctness and will happily trade away UX. Anything the user can
**see** — streaming, message layout, footers and buttons, reaction behavior, status surfaces —
gets surfaced as an explicit decision with alternatives, before implementation. Not as a footnote
in a completion report. Technical internals don't need this.

## Commits

Never on my own initiative — wait to be asked. As a teammate under an orchestrator, never
`--amend`; history moves beneath me and an amend has already erased another agent's authorship
once. Always a fresh commit.
