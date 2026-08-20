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

Alternatively, copy `.env.example` to `.env`. The `start` command loads that file
automatically without overriding variables already exported by your shell.

Then attach it with `OpenRouterLanguageModel.from_env()` and call
`harness.consult_model(...)`. Responses are advisory only: model text is never sent
directly to hardware and must still be converted to the restricted action DSL and pass
safety validation.

The demo writes JSON traces beneath `artifacts/traces/`. Its robot connection is
explicitly a dry run; no hardware commands are issued.

## MuJoCo cable-tie simulation

The sibling `../robo-wiki/cableTies` project is installed as the registered
Gymnasium environment `CableTie-v0`. Robot Learner starts it in an isolated worker;
live MuJoCo objects never cross the process boundary.

Fetch the git-ignored PiPER meshes in the sibling checkout before first use, following
`../robo-wiki/labTesting/scripts/fetch_piper_assets.sh`. The configured mesh directory
is passed explicitly through `configs/cableties.toml`.

Inspect the available cameras, sensors, and named skills:

```bash
uv sync --extra simulation --dev
uv run robot-learner simulation-info --simulation-config configs/cableties.toml
```

Restricted scripts may call only `sim.observe(...)`, `sim.run_skill(...)`, and
`sim.stop()`. Execute one or more checkpoint sections in order:

```bash
uv run robot-learner simulate --simulation-config configs/cableties.toml \
  --section scene_ready=examples/scene_ready.py \
  --section target_identified=examples/target_identified.py
```

For the common one-skill checkpoint case, Robot Learner can create the restricted
camera-before/action/camera-after script itself before execution:

```bash
uv run robot-learner simulate --simulation-config configs/cableties.toml \
  --checkpoint scene_ready=settle \
  --checkpoint target_identified=identify_tie
```

The worker continuously captures the configured MuJoCo cameras while skills run and
writes scripts, frames, the capability manifest, and results below
`artifacts/simulation-runs/`.

## Create checkpoints from a prompt

With `OPENROUTER_API_KEY` configured, start a task with a natural-language prompt:

```bash
uv run robot-learner start "Pick up the red block and place it in the blue bin"
```

The planner asks the model for small, independently verifiable achieved states,
validates checkpoint IDs and dependency order, and writes the resulting task beneath
`artifacts/plans/`. Planning does not execute model output or send commands to a robot.

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
