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

The sibling `../robo-wiki/cableTies` project exposes `CableTieHost`: named
skills, cameras, privileged state (tie poses, hole sites), and MuJoCo
snapshots. Robot Learner talks to that host through `RobotAdapter` in an
isolated worker. Live MuJoCo objects never cross the process boundary.
Gymnasium is optional on the cell package and is not the integration API.

Fetch the git-ignored PiPER meshes in the sibling checkout before first use,
following `../robo-wiki/labTesting/scripts/fetch_piper_assets.sh`. Point
`piper_asset_dir` at `.../piper/assets` (with `piper.xml` as a sibling).
A relative path in `configs/cableties.toml` is a sibling of the
**robotLearner repo** (`../robo-wiki/...`), not of the `configs/` folder.
You can also set `CABLETIES_PIPER_ASSET_DIR`.
`render_backend = "auto"` uses OSMesa on Linux and MuJoCo's native GLFW
default on macOS. Do not set `osmesa` on Mac — that backend is Linux-only.

Inspect the host:

```bash
uv sync --extra simulation --dev
uv run robot-learner simulation-info --simulation-config configs/cableties.toml
```

Explore by calling the adapter (observe / run a skill / snapshot), in argv
order. A successful trace is compiled into a restricted script under the run
directory (`explored.py`). Restricted scripts are the artifact, not the
interaction medium:

```bash
uv run robot-learner simulate --simulation-config configs/cableties.toml \
  --do observe \
  --skill settle \
  --do snapshot \
  --skill identify_tie \
  --do restore=1
```

Replay a saved script, or explore one checkpoint and persist the compiled
camera-before / skill / camera-after script:

```bash
uv run robot-learner simulate --simulation-config configs/cableties.toml \
  --section scene_ready=examples/scene_ready.py \
  --checkpoint target_identified=identify_tie
```

The worker captures configured cameras while skills run and writes scripts,
frames, the capability manifest, and `run.json` below `artifacts/simulation-runs/`.

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
