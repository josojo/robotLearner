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
`harness.consult_model(...)`. `complete(..., images=[...])` attaches local PNG/JPEG
files as OpenRouter image-understanding parts. Responses are advisory only: model
text is never sent directly to hardware and must still be converted to the
restricted action DSL and pass safety validation.

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

## Generate checkpoint scripts with the explorer

`explore` walks a checkpoint plan on one long-lived simulation client. For each
checkpoint it observes privileged state (ties, hole, TCP, phase) and the
configured camera PNGs, asks the language model which catalog
`sim.run_skill(...)` to run, parses the reply with `parse_restricted_script`,
executes that skill, and on failure restores the pre-checkpoint snapshot and
retries up to `max_script_revisions`. Successful transitions are written as
`scripts/<checkpoint>/v1.py`.

The OpenRouter wrapper sends those observe frames as image-understanding
input (`image_url` data URLs). Use a multimodal `OPENROUTER_MODEL`. The
wrapper does not request image generation.

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
uv run robot-learner explore --simulation-config configs/cableties.toml \
  --plan examples/cabletie_plan.json
```

`--plan` is the JSON written by `start` (or the bundled cable-tie example).
Each LLM call may return only one `sim.run_skill` from the host catalog.

## Rewatch a simulation run

Observe frames and in-skill stride captures are stored as
`frames/<camera>/00000000.png` under the run directory. After `explore` or
`simulate`, `review.html` is written there. The HTML player is the replay:
open it in a browser to scrub all cameras together.

```bash
uv run robot-learner review artifacts/simulation-runs/explore_8ebdb612a4c74c818e4d7716766746ad
```

That opens `review.html` and also writes `videos/<camera>.mp4` when `ffmpeg`
is on `PATH` (Homebrew: `brew install ffmpeg`). macOS ffmpeg often lacks
glob input, so encoding uses a concat list. If ffmpeg is missing, Pillow
from the simulation extra writes an animated WebP/GIF instead.
`--no-open` skips the browser; `--no-video` skips encoding.
Failed skill attempts stay in the frame timeline, so you can see the
motion that missed before a restore/retry.

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
