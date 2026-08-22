# Robot Learner

## Project specification and initial engineering north star

**Robot Learner** is a software **learning harness** for developing robust robot task programs from real-world execution. Given a task described in natural language and/or images, the harness decomposes the task into verifiable intermediate checkpoints, retrieves or synthesizes robot programs for those checkpoints, executes them within safety boundaries, evaluates the results, and stores the evidence for future reuse and improvements.

Robot Learner has three complementary design pillars:

1. **Slow Executer.** Initially the program is guided by taking a small action and then sending the newest state to a frontier LLM for future instructions. This flow is slow, but it is how the harness discovers successful executions and safer recoveries.
2. **Dynamic scene-flow decomposition and local reprogramming:** when a scene or action flow fails, split it at an informative point, reconsider the scenes before and after the split, and let the LMM adjust the local flow and instructions.
3. **Continual learning over a library of strategies:** preserve the resulting strategies and their evidence so future executions can reuse what worked and avoid known failures.

The Slow Executer and local reprogramming are the primary mechanisms for *discovering* recoveries in novel scenarios. Once a recovery is validated, it is compiled into the strategy so later executions do not wait for the LMM. The library is what makes those compiled recoveries accumulate into reliability over time.

The central principles are:

> A new program is an additional hypothesis, not a replacement for everything learned before.

> Recovery belongs in the strategy as code, not as a live LMM choice.

The system should preserve successful and unsuccessful strategies, associate them with context, and select the most promising strategy for the current situation. It should also preserve the learned scene boundaries, failure triggers, and recovery flows that determine when the LMM needs to reconsider execution. A strategy that only encodes the happy path, and then asks the LMM which recovery to try after a failed grasp, is incomplete.

## Goals

- Convert a human task description into an executable, inspectable checkpoint graph.
- Support task grounding from text, images, robot observations, or combinations of these.
- Execute first in real environments; simulation may be added as an optional test and data source.
- Make every checkpoint independently verifiable.
- Capture failures as structured traces containing observations, state, actions, compiled recovery, and explanations.
- Compile declared recovery options into executable strategy branches so known local failures do not wait for the LMM.
- Reuse proven strategies before adapting or inventing new ones.
- Compose reusable skills into longer task programs.
- Keep humans in control of safety-critical actions and of approving new recovery branches, not of picking among already-declared recoveries at runtime.
- Make early demos possible with a small number of robot-specific adapters.
- enrich harness by object detection and depth estimations programs based on image information

## Non-goals for the MVP

- Fully autonomous operation around people or unknown hazards.
- A universal robot control abstraction that hides all hardware differences.
- Dependence on repeatable simulation or large offline datasets.
- Automatic deletion or replacement of old strategies.
- Perfect causal explanations of failures. Explanations are hypotheses linked to evidence.
- Using the LMM to choose among recovery options that are already declared on the current strategy.

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

The graph may branch when alternative strategies are available. Recovery for a known local miss lives inside a strategy as compiled branches, not as an extra graph node that waits for the LMM.

### Skill

A reusable capability, such as `detect_loop`, `grasp_point`, `visual_align`, `insert_through`, or `verify_passage`. A skill is an interface and behavioral contract, not necessarily one fixed implementation.

### Strategy / robot program

One concrete implementation of a skill or checkpoint transition. Multiple strategies may coexist for the same checkpoint because different contexts favor different approaches.

A strategy is a compiled program, not a happy-path sequence that later fails into a conversation with the critic. It must include:

- a nominal forward sequence;
- local failure checks (empty gripper, planner error, force limit, stall, verifier fail);
- bounded recovery branches drawn from the checkpoint’s allowed recovery options;
- an escalate condition that stops, captures evidence, and only then requests deliberation.

Recovery options listed on a checkpoint are a safety-bounded menu of *allowed* physical alternatives. They are not a prompt for the LMM at failure time. Before a strategy may run, those options must be compiled into executable recovery branches in the restricted action DSL. `if grasp_failed: retract; reobserve; try next grasp candidate` is strategy code. Asking the LMM which option to pick after the gripper comes up empty is a specification error.

### Strategy library

Persistent storage for strategies and their execution evidence. It must retain old versions, including failures, so the system can learn which approaches work under which conditions.

### Critic

An LMM- or rule-based component that analyzes an execution trace after compiled recovery is exhausted or the failure is unknown. It determines whether the checkpoint was achieved, identifies likely failure causes, and recommends reuse, adaptation, or synthesis. A recommendation to recover is a request to compile a new recovery branch into the next strategy, not a command to pick an option from the current one at runtime.

### Program synthesizer

An LMM- or template-based component that creates a new strategy or adapts an existing one. Generated programs must conform to typed interfaces, declared capabilities, and safety limits before execution, and they must compile the checkpoint’s recovery options into executable recovery branches. A synthesizer output that contains only a happy path is incomplete.

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
               (nominal + compiled recovery)
                    ↓
       Sensors + checkpoint verifier
                    ↓
          Execution trace recorder
          ↙                    ↘
   handled recovery         exhausted / unknown
   → library                → Critic → new strategy
                              (compile a new recovery branch)
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
# executor.run includes nominal actions and compiled recovery branches
trace = executor.run(validated)
result = verifier.evaluate(checkpoint, trace)
if result.passed:
    library.record(strategy, observation, trace, result, critique=None)
    continue
if trace.recovery_exhausted or result.unknown:
    critique = critic.analyze(checkpoint, trace, result)
    strategy = synthesizer.adapt(strategy, critique)  # compile a new recovery branch
    library.record(strategy, observation, trace, result, critique)
```

The real implementation should make each transition observable and interruptible. The critic is not on the hot path of a known local failure. It runs after compiled recovery has nothing left to try, or when the failure is not one of the strategy’s declared checks.

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

The harness should not ask a large LMM to reason continuously. Most of a task should run through a known, validated strategy, including that strategy’s compiled recovery branches. The LMM becomes active at **information-rich moments** where additional reasoning can change the outcome — not at every local miss the strategy already knows how to handle.

These moments are dynamic rather than hard-coded timestamps. The harness detects them from observations, history, and uncertainty:

- a checkpoint is approaching or its preconditions are only partially satisfied;
- visual or state estimates become ambiguous;
- progress stalls or deviates from the expected trajectory *and* the strategy has no remaining recovery branch for that stall;
- a previously observed failure signature is detected that is not already compiled into the current strategy;
- contact, force, occlusion, or object motion differs from the strategy’s context in a way no recovery branch covers;
- compiled recovery branches are exhausted, or a verifier returns an unknown / unhandled failure;
- the scene decomposition itself looks wrong and a cut point is needed, not merely a known local miss;
- the current strategy has low evidence for the present context.

A failed grasp that already has `retry_next_candidate` in the strategy does not trigger deliberation. The executor runs that branch inside the current strategy. Deliberation starts only when those branches are exhausted or the failure is not one of the declared checks. When a deliberation trigger does fire, the harness cuts the long-horizon execution into a local scene:

```text
long-horizon strategy
        ↓
known local failure → run compiled recovery branch → rejoin
        ↓
unhandled event / exhausted recovery / unknown failure
        ↓
stop safely before the known failure boundary
        ↓
capture an evidence bundle
        ↓
historical analysis + LMM deliberation
        ↓
synthesize or adapt a local strategy that compiles a new recovery branch
        ↓
execute and verify the next checkpoint
        ↓
rejoin the long-horizon plan
```

The key distinction is between a **scene segment** and a fixed script step. A segment has:

- a start condition and expected end condition;
- a nominal strategy whose recovery branches are already compiled as code;
- monitored signals and uncertainty thresholds;
- known failure boundaries or trigger signatures, each mapped to a recovery branch or to escalation;
- an evidence-capture policy;
- allowed recovery and replanning actions, executable without a live LMM call until they are exhausted.

This lets the system learn where a segment should be cut. After a failure, the critic should identify the earliest useful intervention point—not merely the final bad frame—and compile a recovery branch or a new cut into the next strategy. On the next attempt, the executor can run the prior segment until just before that point and execute the compiled branch. It should not pause for another model call to rediscover the same recovery.

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

Offline analysis can review full-resolution images, video, state histories, and prior traces without putting real-time latency in the control loop. Its output should update the strategy library with failure signatures, likely causes, useful cut points, and revised strategies whose new recovery branches will run as code on the next attempt.

### Deliberation contract

The LMM should return a structured decision, not unconstrained robot commands:

```yaml
decision: adapt_strategy
reasoning_summary: Align from the side because the loop is partially occluded.
selected_strategy: strategy_024
change: approach_from_left_then_reobserve
cut_point: before_alignment_motion
required_observations: [side_view, loop_orientation, tip_visibility]
compile_recovery_branch:
  on: loop_occluded
  then: [stop, retract_to_safe_pose, observe_side_view, approach_from_left]
  budget: {max_retries: 2}
confidence: 0.78
fallback: stop_for_human_review
```

The safety layer validates the decision and compiles it into the restricted action DSL, including any new recovery branches. The LMM may choose among approved skills, request observations, alter bounded parameters, or propose a new strategy; it may not bypass safety limits, directly issue arbitrary hardware commands, or remain on the runtime path of a recovery the next strategy should already contain.

### Learning without an operator

“Without an operator” should initially mean **no operator choosing the next strategy**, not **no safety supervision**. The first reaction to a known local failure is the strategy’s compiled recovery, not a model call. Only after that code is exhausted does the harness stop at a safe boundary, retrieve historical cases, ask the LMM for a structured local decision, and compile the result into a replacement strategy. Human approval remains required for new risk levels, uncertain safety conditions, or strategies without sufficient evidence.

The learning loop is therefore:

```text
execute a compiled strategy, including its recovery branches
  → on a known local failure, run the matching recovery branch
  → if that recovers, continue and record that the branch helped
  → if recovery is exhausted or the failure is unknown:
        stop at a learned cut point
        capture an evidence bundle
        inspect history offline / on demand
        deliberate over the local scene
        compile a bounded next strategy (nominal + new recovery branches)
        validate, execute, verify
        store what changed and whether it helped
```

This gives the system a long horizon without requiring long-horizon reasoning at every instant. Most execution remains fast and scripted, including scripted recovery. The LMM is used to decide when the current scene decomposition or compiled recovery is no longer sufficient.

## Reliability through checkpoint recovery and local reprogramming

The primary reliability mechanism of Robot Learner is **recoverable execution**. A task does not have to be restarted from the beginning when a strategy fails. It is divided into verified checkpoints, and the robot maintains a bounded recovery position at the latest checkpoint whose success was confirmed.

“Go back to the previous checkpoint” must never mean rewinding time. It means executing a physically valid **recovery policy**. The robot must undo, clear, stabilize, or re-localize the scene using real actions, and it may only retreat when force, contact, and geometry make that safe.

Examples:

- If a pipette misses the tube, retreat, release or regrasp it, and return to the station.
- If a cable tie is misplaced, unlock the fixture, clear it, and select another tie.
- If a filled tube is unstable, place it safely, re-localize it, and retry the return.
- If an insertion partially succeeds, back out only if force and geometry permit it.

When a known local failure fires inside a compiled strategy, the executor does not leave the strategy:

```text
execute strategy for checkpoint C
              ↓
     local failure check fires
              ↓
run the matching recovery branch
  (retract, reobserve, retry declared alternative)
              ↓
        verify C
              ↓
   pass → continue with C → D
   fail and budget remaining → next compiled branch
   fail and budget exhausted or unknown → escalate
```

Only the escalate path may call the LMM, and its job is to write a *new* strategy, not to pick the next runtime action:

```text
stop and preserve failure evidence
              ↓
execute the recovery contract for checkpoint B
              ↓
analyze the failed transition B → C
              ↓
retrieve related strategies and failure cases
              ↓
LMM writes a new bounded approach for B → C
  that includes a compiled recovery branch for this failure
              ↓
validate and execute the new approach
              ↓
verify C and continue with C → D
```

Here, checkpoint `B` is not a physical rewind. It is a **known state from which the transition can be attempted again**. The recovery action might be to return the arm to a saved pose, regrasp an object, clear the workspace, restore object stability, or simply pause and reobserve. Each checkpoint should therefore define:

- a success predicate: how do we know the checkpoint was reached?
- a forward transition: how do we move from this checkpoint to the next one?
- a recovery contract: what physical actions can safely return the scene to a retryable state;
- recovery options: which bounded alternatives a strategy is *allowed* to compile into executable branches;
- a fallback used only after compiled branches are exhausted.

The checkpoint names the allowed recoveries. The strategy compiles them. The executor runs them.

Example checkpoint contract and the strategy that instantiates it:

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

strategy: insert_tip_with_force_limit_v2
checkpoint: aligned_with_hole
nominal:
  - observe
  - align_tip
  - insert_with_force_limit
recovery_branches:
  - on: xy_misalignment
    then: [stop, retract_to_safe_pose, reobserve, adjust_xy, retry_insert]
    budget: {max_retries: 3}
  - on: approach_blocked
    then: [stop, retract_to_safe_pose, change_approach_angle, reobserve]
    budget: {max_retries: 2}
  - on: unknown
    then: [stop, retract_to_safe_pose, reobserve]
    escalate: deliberation
```

The recovery contract is part of the checkpoint’s safety boundary. A strategy may compile only declared recovery options, plus the mandatory safe-stop sequence in the recovery contract. It may not invent a physical undo the checkpoint does not allow. The LMM may propose a new recovery option for review and, if accepted, compile it into a new strategy version. It may not remain in the loop selecting among options that are already declared.

The LMM is not asked to regenerate the entire task. After compiled recovery is exhausted, it analyzes the failed transition, keeps the verified prefix of the task, and produces a replacement strategy for the smallest failing segment — including a new recovery branch so the same miss does not wait for another model call. This limits the search space and preserves everything that already worked.

### Learning from a new scenario

In a genuinely new scenario, the first strategy may fail because the scene differs from the historical context: object pose, occlusion, geometry, lighting, grasp location, or physical interaction may be different. The failure is useful if the harness records:

- the last verified checkpoint;
- the failed checkpoint and attempted strategy;
- the earliest point where the expected scene diverged;
- images and state before, during, and after the failure;
- the observed context and the strategy’s expected context;
- the critic’s hypothesis about the mismatch;
- the replacement strategy and whether it succeeded on retry.

The new strategy is then added beside the old one, tagged with the scenario in which it worked, and it must contain a compiled recovery branch for the miss that just required deliberation. Over time, the library becomes a collection of routes through each checkpoint, rather than a single brittle script. In a future similar scene, retrieval can select the new route immediately and run its recovery as code; in a different scene, the system can again fall back to the last verified checkpoint and adapt locally.

### What the system learns

Robot Learner should learn four related things:

1. **Scene boundaries:** where a long-horizon flow should be split for observation or deliberation.
2. **Local flows:** how to execute the scenes before and after a learned split.
3. **Strategies:** how to move from one checkpoint to the next, including compiled recovery branches for known local failures.
4. **Recovery routes:** how to return to the last useful verified state, and which recovery branches belong in the next strategy rather than in a live LMM decision.

Failure boundaries are part of all four: they describe where the current flow stops being reliable and which new scene boundary, recovery branch, or strategy should be tried. A recovered miss that required deliberation once should, on the next attempt, be a branch already present in the strategy.

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
slow LMM: revise scene flow, interpret history, compile a new strategy
          (nominal actions + recovery branches)
       ↓
compiled recovery in the strategy: known local failures, no model in the loop
       ↓
fast policy / controller: execute time-critical interaction
       ↓
verifier + trace recorder: assess outcome and update memory
```

The middle layer is the answer to ordinary misses — empty grasp, blocked approach, slight misalignment. Those must not wait for image inference. The fast layer remains reserved for contact reflexes that cannot wait even for a compiled branch to finish a retract-and-retry. The slow layer writes new branches; it does not execute them.

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
recovery:
  branches_run: [retry_next_grasp_candidate]
  exhausted: true
  escalated: true
critic:
  likely_causes: [poor_initial_alignment, loop_occluded]
  recommendation: adapt_strategy
  compile_recovery_branch: approach_from_left_then_reobserve
safety_events: []
```

Store enough raw evidence to replay the analysis, while keeping large media in an artifact store referenced by IDs or paths.

## Strategy and contextual history model

An initial relational or document-backed model can use these entities:

```text
Task
  id, description, inputs, constraints

Checkpoint
  id, task_id, name, preconditions, success_predicate, dependencies,
  recovery_contract, recovery_options, fallback

Skill
  id, name, input_schema, output_schema, safety_contract

Strategy
  id, checkpoint_id, skill_ids, program, recovery_branches, version,
  parent_strategy_id

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
class RecoveryBranch:
    on: Predicate
    then: list[Action]
    budget: RecoveryBudget
    escalate: bool = False

class Strategy:
    id: str
    checkpoint_id: str
    required_skills: list[str]
    preconditions: list[Predicate]
    safety_contract: SafetyContract
    actions: list[Action]            # nominal forward sequence
    recovery_branches: list[RecoveryBranch]

    def run(self, context: ExecutionContext) -> ActionResult:
        ...
```

`run` executes the nominal sequence and, on a matching local failure, the corresponding recovery branch, without calling the LMM. A strategy is invalid if it has recovery options on its checkpoint but no compiled branches, or if a branch uses an action outside the checkpoint’s recovery contract.

The executor should expose a small action vocabulary—such as observe, move, grasp, release, open/close gripper, wait, and stop—rather than arbitrary code execution. Programs can be represented as validated action graphs or a restricted DSL before any robot command is sent. Recovery branches are the same DSL: they are part of the program, not a side channel of model text.

## Safe execution boundaries

Safety is a first-class gate, not a post-processing step.

- Require explicit robot connection and task-level authorization.
- Enforce workspace, joint, velocity, acceleration, force, and duration limits.
- Validate preconditions before every strategy, and validate recovery branches against the checkpoint recovery contract.
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
2. The task interpreter proposes a checkpoint graph for human approval, including recovery options per checkpoint.
3. The retriever finds prior strategies for the next checkpoint.
4. The user or LMM selects reuse, bounded adaptation, or synthesis. Synthesis must compile declared recovery options into executable recovery branches.
5. The safety layer validates the action graph, including those branches against the recovery contract.
6. The robot executes at low speed with recording enabled. Known local failures run the compiled branches without a model call.
7. The verifier returns pass, fail, or uncertain.
8. On a handled failure, the executor retries through remaining compiled branches and records which branch ran.
9. Only if those branches are exhausted or the failure is unknown does the recovery manager return to the latest verified checkpoint, and the critic propose a replacement strategy that includes a new compiled recovery branch.
10. The library stores the full result, including the failed strategy, the branches that ran, and the replacement strategy, for later runs.

For the first implementation, it is acceptable for a human to approve the graph, the strategy, and the compiled recovery branches. The learning harness should make those decisions explicit and recordable before attempting to automate them. The thing that must not wait for a later phase is the rule itself: recovery that is already declared must run as code.

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

Define schemas, the restricted action DSL, robot adapter, artifact recorder, emergency stop, and one manually authored strategy that includes at least one compiled recovery branch. Run one checkpoint end-to-end, trigger the local failure, and inspect a trace in which recovery ran without a model call.

### Phase 2 — Checkpoint graph and verification

Add task parsing, graph proposal/editing, checkpoint-specific verifiers, recovery contracts, and human approval gates. Make pass/fail/uncertain outcomes reliable before adding autonomous synthesis. Reject a strategy that lists recovery options but does not compile them.

### Phase 3 — Persistent strategy library

Add versioned strategies, contextual execution history, retrieval, ranking, and strategy comparison. Demonstrate that an older successful strategy remains available after a new strategy fails, including the recovery branches that distinguished them.

### Phase 4 — Critic and bounded adaptation

Use the LMM to summarize failures from traces and propose narrowly scoped changes. Those changes must compile into the next strategy’s nominal sequence and recovery branches, not into a one-shot runtime command. Validate all generated programs against schemas, safety contracts, and the checkpoint recovery contract.

### Phase 5 — Learned scene cutting and autonomous recovery

Detect failure precursors, learn intervention boundaries, capture evidence bundles, and retrieve similar historical cases. When compiled recovery is exhausted, the LMM writes a new local strategy — including a new recovery branch — without operator selection of the next action. It does not sit in the loop choosing among options the current strategy already declared. Keep safety approval gates for novel or uncertain actions.

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

Checkpoints: detect hole → align axes → approach → insert with bounded force → verify seating. Use force/torque limits and a hard stop. The insert strategy must compile `retract_and_adjust_xy` as a recovery branch so a miss retries without a model call; only a second, unknown failure should reach the critic.

## Evaluation metrics

- Checkpoint success rate, overall task success rate, and recovery rate.
- Share of recoveries handled by compiled strategy branches versus recoveries that required LMM deliberation.
- Number of real-world trials needed to reach a target success rate.
- Strategy reuse rate versus newly synthesized strategies.
- Performance by context, not only aggregate performance.
- Safety stops, near misses, and human interventions.
- Quality and completeness of failure traces.
- Whether old strategies remain useful after new variants are introduced.

## Design rule of thumb

When a known local failure fires, the system should not ask the LMM what to do next. It should run the recovery already compiled into the strategy.

When that code is exhausted, the system should not merely ask, “What is the new script?” It should ask:

> What happened, which prior strategies are relevant, what evidence supports reuse or adaptation, and what bounded experiment is safe to compile into the next strategy — including the recovery branch that will handle this miss without another model call?

That question is the heart of the Robot Learner learning harness.

