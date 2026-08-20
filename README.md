# Robot Learner

Robot Learner is an initial, safety-first Python framework for learning robot task
programs from checkpoint-based executions. It turns a validated strategy into a small
action vocabulary, checks it against explicit limits, executes it through a robot
adapter, verifies the result, and preserves an immutable trace.

This scaffold implements the Phase 1/MVP seam from `SPECIFICATION.md`. It deliberately
does not connect to real hardware. Model deliberation is available through OpenRouter,
while the included CLI uses a deterministic dry-run adapter.

## Getting started

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --dev
uv run robot-learner demo --config configs/default.toml
uv run pytest
uv run ruff check .
uv run mypy
```

## OpenRouter

The harness accepts an `OpenRouterLanguageModel` through its optional `language_model`
dependency. Configure the official OpenRouter SDK with environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENROUTER_MODEL="openai/gpt-4o-mini"  # optional
```

Alternatively, copy `.env.example` to `.env` and load it into your shell or process
manager. The application does not automatically load `.env` files.

Then attach it with `OpenRouterLanguageModel.from_env()` and call
`harness.consult_model(...)`. Responses are advisory only: model text is never sent
directly to hardware and must still be converted to the restricted action DSL and pass
safety validation.

The demo writes JSON traces beneath `artifacts/traces/`. Its robot connection is
explicitly a dry run; no hardware commands are issued.

## Architecture

- `models`: immutable task, checkpoint, strategy, action, safety, and trace schemas.
- `safety`: validates authorization, trust, action vocabulary, and numeric limits.
- `executor`: an interruptible runtime plus a narrow robot adapter protocol.
- `verification`: independent checkpoint verifier contracts.
- `tracing`: append-only JSON artifact recording.
- `library`: SQLite strategy metadata and contextual execution history.
- `harness`: the observable validate → execute → verify → record workflow.

Planning, perception, synthesis, and deliberation are represented by extension
protocols in `ports.py`. A production adapter should translate the restricted action
DSL into vendor commands and retain a hardware emergency stop below this process.

## Safety assumptions

The default configuration rejects untrusted strategies, requires explicit task
authorization, limits motion speed/force/duration, and stops on lost observations.
Configuration is a deployment convenience, not a substitute for certified robot-side
safety controls. Review limits and implement physical interlocks before real use.

