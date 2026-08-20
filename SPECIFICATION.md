# Robot Learner

## Project specification and initial engineering north star

**Robot Learner** is a software **learning harness** for developing robust robot task programs from real-world execution. Given a task described in natural language and/or images, the harness decomposes the task into verifiable intermediate checkpoints, retrieves or synthesizes robot programs for those checkpoints, executes them within safety boundaries, evaluates the results, and stores the evidence for future reuse and improvements.

Robot Learner has two complementary design pillars:

1. **Slow Executer** Initially the program is guided by a taking a small action and then sending the newest state to a frontier LLM for future instructions. While this flow is super slow, it allows to learn successful executions and safer.
1. **Dynamic scene-flow decomposition and local reprogramming:** when a scene or action flow fails, split it at an informative point, reconsider the scenes before and after the split, and let the LMM adjust the local flow and instructions.
2. **Continual learning over a library of strategies:** preserve the resulting strategies and their evidence so future executions can reuse what worked and avoid known failures.

The first pillar is the primary mechanism for handling novel scenarios. The second pillar makes those repairs accumulate into reliability over time.

The central principle is:

> A new program is an additional hypothesis, not a replacement for everything learned before.

The system should preserve successful and unsuccessful strategies, associate them with context, and select the most promising strategy for the current situation. It should also preserve the learned scene boundaries, failure triggers, and recovery flows that determine when the LMM needs to reconsider execution.

## Goals

- Convert a human task description into an executable, inspectable checkpoint graph.
- Support task grounding from text, images, robot observations, or combinations of these.
- Execute first in real environments; simulation may be added as an optional test and data source.
- Make every checkpoint independently verifiable.
- Capture failures as structured traces containing observations, state, actions, and explanations.
- Reuse proven strategies before adapting or inventing new ones.
- Compose reusable skills into longer task programs.
- Keep humans in control of safety-critical actions and recovery decisions.
- Make early demos possible with a small number of robot-specific adapters.
- enrich harness by object detection and depth estimations programs based on image information

## Non-goals for the MVP

- Fully autonomous operation around people or unknown hazards.
- A universal robot control abstraction that hides all hardware differences.
- Dependence on repeatable simulation or large offline datasets.
- Automatic deletion or replacement of old strategies.
- Perfect causal explanations of failures. Explanations are hypotheses linked to evidence.

## Core concepts

### Task

A task is the desired outcome plus its environmental context.

```yaml
task_id: thread_string_through_loop
goal: Thread the end of the string through the loop.
inputs:
  text: "Thread the loose end through the metal loop."
  images: [scene_001.jpg]
constraints:
  - Keep the object on the work surface.
  - Stop if the gripper loses the string.
```

### Checkpoint graph

A directed graph of intermediate states that must be achieved and verified. Nodes describe observable conditions; edges describe transitions or robot programs.

Example:

```text
identify_string_and_loop
        ↓
grasp_string_end
        ↓
align_string_with_loop
        ↓
pass_string_through_loop
        ↓
release_and_verify
```

The graph may branch when alternative strategies or recovery actions are available.

### Skill

A reusable capability, such as `detect_loop`, `grasp_point`, `visual_align`, `insert_through`, or `verify_passage`. A skill is an interface and behavioral contract, not necessarily one fixed implementation.

### Strategy / robot program

One concrete implementation of a skill or checkpoint transition. Multiple strategies may coexist for the same checkpoint because different contexts favor different approaches.

### Strategy library

Persistent storage for strategies and their execution evidence. It must retain old versions, including failures, so the system can learn which approaches work under which conditions.

### Critic

An LMM- or rule-based component that analyzes an execution trace, determines whether the checkpoint was achieved, identifies likely failure causes, and recommends reuse, adaptation, recovery, or synthesis.

### Program synthesizer

An LMM- or template-based component that creates a new strategy or adapts an existing one. Generated programs must conform to typed interfaces, declared capabilities, and safety limits before execution.

## System architecture

```text
Task input: text / images / observations
                    ↓
             Task interpreter
                    ↓
          Checkpoint graph builder
                    ↓
       Strategy retriever and ranker
          ↙          ↓          ↘
       reuse      adapt       synthesize
          ↘          ↓          ↙
             Safety validator
                    ↓
             Robot executor
                    ↓
       Sensors + checkpoint verifier
                    ↓
          Execution trace recorder
                    ↓
        Critic → library and next step
```

Suggested modules:

- `task_model`: task, object, constraint, and checkpoint schemas.
- `perception`: image/state observations and object grounding.
- `planner`: checkpoint graph creation and transition selection.
- `library`: strategy storage, retrieval, versioning, and contextual scoring.
- `synthesizer`: strategy generation and adaptation.
- `safety`: permissions, workspace limits, force/velocity limits, timeouts, and human approval gates.
- `executor`: robot-specific action runtime.
- `verification`: checkpoint-specific success predicates.
- `tracing`: immutable observations, actions, outcomes, and failure evidence.
- `deliberation`: event-triggered scene cutting, evidence review, and long-horizon decisions.
- `demo_runner`: repeatable command-line or notebook demos.

## Execution policy

For every checkpoint, use this order:

1. **Reuse** a previous strategy when its context and historical evidence are sufficiently relevant.
2. **Adapt** a previous strategy when it is close to the current context but needs a bounded change.
3. **Invent** a new strategy only when reuse and adaptation are unavailable or poorly supported.

After execution, retain the strategy and append a new result. Never overwrite the strategy’s history. A strategy’s score should consider both success rate and contextual similarity, not only its global average.

Conceptually:

```python
strategy = library.select(checkpoint, observation)
if strategy is None:
    strategy = synthesizer.create(checkpoint, observation, library.related(checkpoint))

validated = safety.validate(strategy, observation)
trace = executor.run(validated)
result = verifier.evaluate(checkpoint, trace)
critique = critic.analyze(checkpoint, trace, result)
library.record(strategy, observation, trace, result, critique)
```

The real implementation should make each transition observable and interruptible.

## Dynamic scene-flow decomposition

Robot Learner should not treat a task as one fixed script or as a permanently fixed sequence of scenes. It begins with a proposed long-horizon flow, but execution can expose that the decomposition is wrong: a scene may be too large, a transition may require an intermediate observation, or a supposedly simple action may contain the real difficulty.

When this happens, the harness splits the flow at an informative point and reopens the local plan:

```text
scene A → scene B → scene C
             ↓ failure or mismatch
scene A → scene B₁ → deliberation → scene B₂ → scene C
```

The split should produce new instructions for both sides of the boundary:

- how to execute the scene before the split and stop at a better state;
- what evidence to capture at that state;
- how to interpret the state and choose the next action;
- how to execute the revised scene after the split;
- how to verify the transition and recover if it fails again.

This is more than inserting an extra checkpoint. It allows the LMM to revise the local flow itself: add an observation, change the order of actions, insert a regrasp, alter the approach direction, or replace one transition with several smaller transitions. The global task plan remains intact wherever its verified prefix and suffix are still valid.

## Event-triggered deliberation

The harness should not ask a large LMM to reason continuously. Most of a task should run through a known, validated strategy. The LMM becomes active at **information-rich moments** where additional reasoning can change the outcome.

These moments are dynamic rather than hard-coded timestamps. The harness detects them from observations, history, and uncertainty:

- a checkpoint is approaching or its preconditions are only partially satisfied;
- visual or state estimates become ambiguous;
- progress stalls or deviates from the expected trajectory;
- a previously observed failure signature is detected;
- contact, force, occlusion, or object motion differs from the strategy’s context;
- a verifier returns failure or uncertainty;
- the current strategy has low evidence for the present context.

At such a point, the harness cuts the long-horizon execution into a local scene:

```text
long-horizon strategy
        ↓
event / failure signature detected
        ↓
stop safely before the known failure boundary
        ↓
capture an evidence bundle
        ↓
historical analysis + LMM deliberation
        ↓
reuse, adapt, recover, or synthesize a local strategy
        ↓
execute and verify the next checkpoint
        ↓
rejoin the long-horizon plan
```

The key distinction is between a **scene segment** and a fixed script step. A segment has:

- a start condition and expected end condition;
- a nominal strategy;
- monitored signals and uncertainty thresholds;
- known failure boundaries or trigger signatures;
- an evidence-capture policy;
- allowed recovery and replanning actions.

This lets the system learn where a segment should be cut. After a failure, the critic should identify the earliest useful intervention point—not merely the final bad frame. On the next attempt, the executor can run the prior segment until just before that point, pause, collect higher-quality observations, and deliberate with the relevant historical cases.

### Evidence bundles for deliberation

When a deliberation trigger fires, capture a bounded bundle rather than the entire lifetime history:

```yaml
evidence_bundle:
  trigger: prior_failure_signature
  checkpoint_id: align_string_with_loop
  frames: [frame_before_01.jpg, frame_before_02.jpg, frame_trigger.jpg]
  temporal_window: {before_s: 3.0, after_s: 0.5}
  robot_state: state_snapshot.json
  action_history: actions_since_segment_start.json
  scene_state: {loop_pose: ..., string_tip_pose: ..., visibility: 0.62}
  retrieved_cases: [execution_017, execution_024]
  uncertainty: {loop_pose: 0.31, string_tip_pose: 0.44}
```

Offline analysis can review full-resolution images, video, state histories, and prior traces without putting real-time latency in the control loop. Its output should update the strategy library with failure signatures, likely causes, useful cut points, and revised strategies.

### Deliberation contract

The LMM should return a structured decision, not unconstrained robot commands:

```yaml
decision: adapt_strategy
reasoning_summary: Align from the side because the loop is partially occluded.
selected_strategy: strategy_024
change: approach_from_left_then_reobserve
cut_point: before_alignment_motion
required_observations: [side_view,_loop_orientation,tip_visibility]
confidence: 0.78
fallback: stop_for_human_review
```

The safety layer validates the decision and compiles it into the restricted action DSL. The LMM may choose among approved skills, request observations, alter bounded parameters, or propose a new strategy; it may not bypass safety limits or directly issue arbitrary hardware commands.

### Learning without an operator

“Without an operator” should initially mean **no operator choosing the next strategy**, not **no safety supervision**. The harness can autonomously detect a failure, stop at a safe boundary, retrieve historical cases, ask the LMM for a structured local decision, execute a validated recovery, and record the result. Human approval remains required for new risk levels, uncertain safety conditions, or strategies without sufficient evidence.

The learning loop is therefore:

```text
execute mostly scripted behavior
  → detect deviation or known failure precursor
  → stop at learned cut point
  → inspect history offline / on demand
  → deliberate over the local scene
  → execute a bounded next move
  → verify
  → store what changed and whether it helped
```

This gives the system a long horizon without requiring long-horizon reasoning at every instant. Most execution remains fast and scripted; the LMM is used to decide when the current scene decomposition or action flow needs to change.

## Reliability through checkpoint recovery and local reprogramming

The primary reliability mechanism of Robot Learner is **recoverable execution**. A task does not have to be restarted from the beginning when a strategy fails. It is divided into verified checkpoints, and the robot maintains a bounded recovery position at the latest checkpoint whose success was confirmed.

“Go back to the previous checkpoint” must never mean rewinding time. It means executing a physically valid **recovery policy**. The robot must undo, clear, stabilize, or re-localize the scene using real actions, and it may only retreat when force, contact, and geometry make that safe.

Examples:

- If a pipette misses the tube, retreat, release or regrasp it, and return to the station.
- If a cable tie is misplaced, unlock the fixture, clear it, and select another tie.
- If a filled tube is unstable, place it safely, re-localize it, and retry the return.
- If an insertion partially succeeds, back out only if force and geometry permit it.

When execution enters a scene that the current script cannot solve:

```text
execute strategy for checkpoint C
              ↓
        strategy fails
              ↓
stop and preserve failure evidence
              ↓
execute the recovery contract for checkpoint B
              ↓
analyze the failed transition B → C
              ↓
retrieve related strategies and failure cases
              ↓
LMM writes a new bounded approach for B → C
              ↓
validate and execute the new approach
              ↓
verify C and continue with C → D
```

Here, checkpoint `B` is not a physical rewind. It is a **known state from which the transition can be attempted again**. The recovery action might be to return the arm to a saved pose, regrasp an object, clear the workspace, restore object stability, or simply pause and reobserve. Each checkpoint should therefore define both:

- a success predicate: how do we know the checkpoint was reached?
- a forward transition: how do we move from this checkpoint to the next one?
- a recovery contract: what physical actions can safely return the scene to a retryable state?
- recovery options: which bounded alternatives may be attempted after analysis?

Example checkpoint contract:

```yaml
checkpoint: aligned_with_hole
success_predicate: tip_is_aligned
forward_transition:
  strategy: insert_tip_with_force_limit
recovery_contract:
  preconditions: [contact_force_below_safe_threshold, retreat_path_clear]
  actions:
    - stop
    - retract_to_safe_pose
    - release_if_holding
    - reobserve
recovery_options:
  - adjust_xy
  - change_approach_angle
  - change_grasp_point
  - discard_and_restart
fallback: stop_for_human_review
```

The recovery contract is part of the checkpoint’s safety boundary. The LMM can select among declared recovery options or propose a new one for review, but it cannot assume that an object, fixture, or robot can be restored automatically.

The LMM is not asked to regenerate the entire task. It analyzes the failed transition, keeps the verified prefix of the task, and produces a replacement strategy for the smallest failing segment. This limits the search space and preserves everything that already worked.

### Learning from a new scenario

In a genuinely new scenario, the first strategy may fail because the scene differs from the historical context: object pose, occlusion, geometry, lighting, grasp location, or physical interaction may be different. The failure is useful if the harness records:

- the last verified checkpoint;
- the failed checkpoint and attempted strategy;
- the earliest point where the expected scene diverged;
- images and state before, during, and after the failure;
- the observed context and the strategy’s expected context;
- the critic’s hypothesis about the mismatch;
- the replacement strategy and whether it succeeded on retry.

The new strategy is then added beside the old one, tagged with the scenario in which it worked. Over time, the library becomes a collection of routes through each checkpoint, rather than a single brittle script. In a future similar scene, retrieval can select the new route immediately; in a different scene, the system can again fall back to the last verified checkpoint and adapt locally.

### What the system learns

Robot Learner should learn four related things:

1. **Scene boundaries:** where a long-horizon flow should be split for observation or deliberation.
2. **Local flows:** how to execute the scenes before and after a learned split.
3. **Strategies:** how to move from one checkpoint to the next.
4. **Recovery routes:** how to return to the last useful verified state and try another strategy.

Failure boundaries are part of all four: they describe where the current flow stops being reliable and which new scene boundary, recovery action, or strategy should be tried.

This is the reliability advantage over a monolithic task script: an error in one transition does not invalidate the verified task prefix or force the robot to rediscover the whole task.

## Fast interaction policies

Some actions cannot wait for normal LMM inference over images. Examples include catching a slipping object, maintaining contact during insertion, reacting to force spikes, or following a moving target. These actions should use a fast local policy, controller, or later a custom fine-tuned vision-language-action (VLA) model.

The learning harness should still manage these policies at the scene level:

- the LMM defines when the fast policy should be invoked;
- the policy receives a bounded observation and a clear goal;
- safety controllers enforce force, velocity, and workspace limits;
- the harness verifies the result;
- failures and sensor traces are stored for later analysis and policy improvement.

This creates a layered architecture rather than requiring one model to do everything:

```text
slow LMM: revise scene flow, interpret history, select strategy
       ↓
fast policy / controller: execute time-critical interaction
       ↓
verifier + trace recorder: assess outcome and update memory
```

Initially, the fast layer can be a hand-designed controller or an existing policy. Later, repeated successful traces and failure cases can support custom fine-tuning for the robot, task family, or interaction mode.

## Verification and failure traces

Checkpoint verification should be explicit and independent from the program that attempted the checkpoint. A verifier may use geometry, force/torque, joint state, object tracking, image comparison, or an LMM grounded in captured evidence.

Every execution should produce an immutable trace containing at least:

```yaml
execution_id: exec_2026_001
task_id: thread_string_through_loop
checkpoint_id: pass_string_through_loop
strategy_id: strategy_017
start_observation:
  image: scene_before.jpg
  robot_state: state_before.json
actions:
  - timestamp: 12.4
    command: move_cartesian
    parameters: {target: [0.31, -0.08, 0.22], speed: 0.05}
observations:
  - timestamp: 13.1
    image: scene_after_01.jpg
    robot_state: state_after_01.json
outcome: failure
verification:
  passed: false
  reason: string_tip_missed_loop
critic:
  likely_causes: [poor_initial_alignment,_loop_occluded]
  recommendation: adapt_strategy
safety_events: []
```

Store enough raw evidence to replay the analysis, while keeping large media in an artifact store referenced by IDs or paths.

## Strategy and contextual history model

An initial relational or document-backed model can use these entities:

```text
Task
  id, description, inputs, constraints

Checkpoint
  id, task_id, name, preconditions, success_predicate, dependencies

Skill
  id, name, input_schema, output_schema, safety_contract

Strategy
  id, checkpoint_id, skill_ids, program, version, parent_strategy_id

Execution
  id, strategy_id, observation_signature, started_at, duration, outcome

Evidence
  execution_id, images, state_snapshots, action_log, sensor_artifacts

Critique
  execution_id, likely_causes, confidence, recommended_change
```

`observation_signature` should capture useful context, such as object pose, size, orientation, visibility, grasp location, workspace configuration, and robot identity. It can begin as structured metadata and later include learned embeddings.

At minimum, retrieve strategies by checkpoint and rank them using:

```text
score = contextual_similarity
      × contextual_success_rate
      × safety_compatibility
      × recency_factor
```

Do not let recency erase older strategies. Keep global and context-specific statistics, including successes, failures, attempted count, and confidence.

## Program interface

Keep generated programs constrained and typed. A strategy should declare:

```python
class Strategy:
    id: str
    checkpoint_id: str
    required_skills: list[str]
    preconditions: list[Predicate]
    safety_contract: SafetyContract

    def run(self, context: ExecutionContext) -> ActionResult:
        ...
```

The executor should expose a small action vocabulary—such as observe, move, grasp, release, open/close gripper, wait, and stop—rather than arbitrary code execution. Programs can be represented as validated action graphs or a restricted DSL before any robot command is sent.

## Safe execution boundaries

Safety is a first-class gate, not a post-processing step.

- Require explicit robot connection and task-level authorization.
- Enforce workspace, joint, velocity, acceleration, force, and duration limits.
- Validate preconditions before every strategy.
- Support immediate stop, watchdog timeouts, and lost-observation handling.
- Require human approval for initially untrusted strategies or hazardous actions.
- Separate planning/synthesis from actuation permissions.
- Log every command and safety event.
- Default to dry-run or low-speed mode when a strategy has no evidence.
- Treat verifier uncertainty as a reason to stop or request inspection.

## Simulation

Simulation is optional. The architecture must work when a real-world experiment cannot be repeated. If simulation is available, use it for preflight checks, broad strategy exploration, regression tests, and synthetic traces—but store simulated and real evidence separately and do not assume simulation success transfers directly to hardware.

## MVP architecture

Start with one robot, one camera configuration, a small action DSL, filesystem-backed artifacts, and a lightweight database such as SQLite.

MVP flow:

1. User supplies a task description and an initial image.
2. The task interpreter proposes a checkpoint graph for human approval.
3. The retriever finds prior strategies for the next checkpoint.
4. The user or LMM selects reuse, bounded adaptation, or synthesis.
5. The safety layer validates the action graph.
6. The robot executes at low speed with recording enabled.
7. The verifier returns pass, fail, or uncertain.
8. On failure, the recovery manager returns to the latest verified checkpoint when safe.
9. The critic analyzes the failed transition and the synthesizer proposes a local replacement strategy.
10. The library stores the full result, including the failed and replacement strategies, for later runs.

For the first implementation, it is acceptable for a human to approve the graph, strategy, and recovery action. The learning harness should make those decisions explicit and recordable before attempting to automate them.

## Suggested repository layout

```text
robot-learner/
├── README.md
├── pyproject.toml
├── configs/
├── robot_learner/
│   ├── models/
│   ├── task_model/
│   ├── perception/
│   ├── planner/
│   ├── library/
│   ├── synthesizer/
│   ├── safety/
│   ├── executor/
│   ├── verification/
│   └── tracing/
├── strategies/
│   └── <checkpoint-id>/
├── artifacts/
│   ├── images/
│   ├── states/
│   └── traces/
├── demos/
├── tests/
└── docs/
```

## Implementation phases

### Phase 1 — Instrumented execution

Define schemas, the restricted action DSL, robot adapter, artifact recorder, emergency stop, and one manually authored strategy. Run one checkpoint end-to-end and inspect its trace.

### Phase 2 — Checkpoint graph and verification

Add task parsing, graph proposal/editing, checkpoint-specific verifiers, and human approval gates. Make pass/fail/uncertain outcomes reliable before adding autonomous synthesis.

### Phase 3 — Persistent strategy library

Add versioned strategies, contextual execution history, retrieval, ranking, and strategy comparison. Demonstrate that an older successful strategy remains available after a new strategy fails.

### Phase 4 — Critic and bounded adaptation

Use the LMM to summarize failures from traces and propose narrowly scoped changes. Validate all generated programs against schemas and safety contracts.

### Phase 5 — Learned scene cutting and autonomous recovery

Detect failure precursors, learn intervention boundaries, capture evidence bundles, retrieve similar historical cases, and let the LMM select a validated local recovery without operator selection. Keep safety approval gates for novel or uncertain actions.

### Phase 6 — Composition and invention

Represent skills independently, compose them into checkpoint programs, and synthesize new strategies only when retrieval and adaptation are insufficient.

### Phase 7 — Optional simulation and evaluation

Add simulation adapters, regression suites, and metrics comparing reuse, adaptation, and invention across real and simulated contexts.

## Initial demos

### 1. Thread string through a loop

Checkpoints: detect string and loop → grasp string end → align → pass through → verify. Begin with one visual alignment strategy, then intentionally vary orientation or occlusion to create multiple strategies and show contextual retrieval.

### 2. Pick and place an object into a marked region

Checkpoints: identify object → grasp → transport → place → verify containment. Demonstrate that a slow approach strategy is retained alongside a faster strategy that only works when the object is well aligned.

### 3. Insert a peg into a hole or connector

Checkpoints: detect hole → align axes → approach → insert with bounded force → verify seating. Use force/torque limits and a hard stop to demonstrate safe failure handling and trace-based adaptation.

## Evaluation metrics

- Checkpoint success rate, overall task success rate, and recovery rate.
- Number of real-world trials needed to reach a target success rate.
- Strategy reuse rate versus newly synthesized strategies.
- Performance by context, not only aggregate performance.
- Safety stops, near misses, and human interventions.
- Quality and completeness of failure traces.
- Whether old strategies remain useful after new variants are introduced.

## Design rule of thumb

When the robot fails, the system should not merely ask, “What is the new script?” It should ask:

> What happened, which prior strategies are relevant, what evidence supports reuse or adaptation, and what bounded experiment is safe to try next?

That question is the heart of the Robot Learner learning harness.

