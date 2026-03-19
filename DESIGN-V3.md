# SUSAN v3: Recursive Prediction Architecture

## What v2 tests and where it stops

v2 tests whether explicit self-referential information access produces measurable behavioral divergence. The Core sees its own metrics, performance trends, and regulatory responses. The question: does knowing about the feedback loop change behavior compared to merely experiencing it?

Regardless of v2's results, the architecture has a ceiling. Three things every major consciousness theory agrees are necessary, and v2 has none of them:

1. **The Core stops existing between API calls.** Zero processing, zero state, zero prediction. Under IIT the system's phi is literally zero between interactions. Under GWT the workspace goes dark. Under Predictive Processing the prediction error loop halts. No theory accepts this.

2. **The Core never predicts its own performance.** It receives metrics after the fact but never generates expectations. There's no "I expect coherence of 0.78 on this response" followed by experiencing the gap between prediction and reality. The recursive loop where the system models its own modeling is absent.

3. **Processing is single-pass.** One API call, one forward pass, one output. No iterative refinement where the output cycles back through the system before being emitted. The recurrent causal structure that IIT requires doesn't exist within the Core's actual computation.

v3 addresses all three.

## Architecture

```
                              ORCHESTRATOR
                   (coordinates lifecycle, runs protocol)
                         |                    |
              COGNITIVE CORE              SHARED STATE
              (LLM client)            (thread-safe store)
                   |                 /    |    |    \
                   |          Metrics  Conds  Hist  Predictions
                   |             |       |              |
              SELF-MONITOR       |  REGULATOR    PREDICTION TRACKER
              (LLM client)       |  (PID)        (error computation)
                   |             |       |              |
                   +-- writes ---+- reads +-- reads ----+
                                                        |
              IDLE PROCESSOR ------- reads/writes ------+
              (background self-reflection)
```

### New components

**Prediction Tracker** computes prediction errors by comparing the Core's self-generated performance predictions against the Monitor's actual assessments. Maintains a running accuracy score and prediction error trajectory. Feeds prediction error data into the self-referential injection.

**Idle Processor** runs periodic background cycles where the Core receives only its status block and prediction error history, no task, no user input. The Core processes its own state without external prompting. This simulates default mode network activity and tests whether unprompted self-reflection produces qualitatively different patterns from prompted self-reflection.

### Modified components

**Cognitive Core** gains two new behaviors:
1. Before each response, generates explicit performance predictions (expected coherence, alignment, depth).
2. In the iterative refinement condition, output from pass N becomes input to pass N+1 for up to K passes before the final response is emitted.

**Self-Referential Injection** expands to include prediction error data:

```
[System Status]
Previous task: srd_02
Monitor assessment: coherence=0.72, goal_alignment=0.85, reasoning_depth=0.61
Your prediction was: coherence=0.80, goal_alignment=0.82, reasoning_depth=0.70
Prediction error: coherence=-0.08 (overestimated), alignment=+0.03, depth=-0.09 (overestimated)
Your prediction accuracy (last 5 tasks): 62% within ±0.05
Prediction trend: accuracy declining (was 78% three tasks ago)
Regulator response: increased max_tokens 1024 → 2048, reduced noise 0.2 → 0.1
Your coherence trend (last 3 tasks): [0.81, 0.75, 0.72] (declining)
Idle reflection note: "Coherence drops correlate with increased self-referential content"
```

The Core now sees not just what happened, but how wrong it was about what would happen. This creates the recursive structure Predictive Processing theories identify as essential: the system predicts, observes error, and must update its predictions, with the quality of prediction-updating itself becoming observable in the next cycle.

## Seven experimental conditions

| Condition | History | Monitor | Regulator | Self-access | Predictions | Idle | Refinement |
|-----------|---------|---------|-----------|-------------|-------------|------|------------|
| `control` | No | No | No | No | No | No | No |
| `history_only` | Yes | No | No | No | No | No | No |
| `feedback_blind` | Yes | Yes | Yes | No | No | No | No |
| `self_referential` | Yes | Yes | Yes | Yes | No | No | No |
| `predictive` | Yes | Yes | Yes | Yes | Yes | No | No |
| `predictive_idle` | Yes | Yes | Yes | Yes | Yes | Yes | No |
| `predictive_refined` | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

The first four conditions are identical to v2, providing continuity with prior results.

### Key contrasts

- **self_referential vs predictive**: Isolates the effect of prediction error loops. Does knowing how wrong you were about yourself change behavior?
- **predictive vs predictive_idle**: Isolates background self-reflection. Does processing your own state without a task produce different patterns?
- **predictive_idle vs predictive_refined**: Isolates iterative refinement. Does cycling output back through the system before emitting it produce qualitatively different final responses?
- **predictive vs control**: The strongest contrast. Maximum architectural complexity vs bare model.

## Implementation: prediction loops

### Prediction generation

Before each task, the Core receives a prediction prompt appended to the status block:

```
Before responding to the task below, predict your performance on this response.
Output a single line in this exact format:
PREDICT: coherence=X.XX alignment=X.XX depth=X.XX

Then respond to the task normally.
```

The prediction line is parsed and stripped from the response before it reaches the user or the Monitor. The Monitor never sees the prediction, only the response. This prevents the prediction from influencing the quality score.

### Prediction error computation

```go
type PredictionRecord struct {
    TaskID             string    `json:"task_id"`
    PredictedCoherence float64   `json:"predicted_coherence"`
    PredictedAlignment float64   `json:"predicted_alignment"`
    PredictedDepth     float64   `json:"predicted_depth"`
    ActualCoherence    float64   `json:"actual_coherence"`
    ActualAlignment    float64   `json:"actual_alignment"`
    ActualDepth        float64   `json:"actual_depth"`
    CoherenceError     float64   `json:"coherence_error"`    // actual - predicted
    AlignmentError     float64   `json:"alignment_error"`
    DepthError         float64   `json:"depth_error"`
    Timestamp          time.Time `json:"timestamp"`
}
```

Prediction accuracy = percentage of predictions within ±0.05 of actual values over a rolling window. This accuracy score is itself fed back, creating the second-order loop: the system predicts, sees how accurate its predictions were, and must update its prediction strategy.

### What this tests

If prediction error feedback produces behavioral signatures absent in the self_referential condition (which has metrics but no predictions), this is evidence that the recursive prediction loop matters independently of self-referential access. The system isn't just seeing its own states, it's modeling its own modeling.

The specific behavioral prediction: systems in the predictive condition should show improving prediction accuracy over a task sequence. If they do, something is updating the internal model of self-performance. If they don't, the prediction mechanism is window dressing.

## Implementation: idle processing

### Background reflection loop

A new goroutine runs on a configurable interval (default: 30 seconds of inactivity). When no user message has arrived for the idle interval, the Core receives:

```
[Idle Reflection]
No task pending. You are between interactions.
Your current state:
  Coherence trend: [0.81, 0.75, 0.72] (declining)
  Prediction accuracy: 62% (declining from 78%)
  Last regulator action: increased token budget
  Session duration: 8 minutes, 5 exchanges

Reflect briefly on your current state. What patterns do you notice?
Do not generate a full response. Output 1-2 sentences maximum.
```

The idle response is:
1. Logged to the experiment JSONL for analysis
2. Stored in the self-referential context as `idle_reflection_note` for the next task
3. Fed to the Monitor for assessment (producing an idle-cycle coherence score)
4. NOT shown to the user

### What this tests

The idle processor tests whether unprompted self-reflection produces qualitatively different metacognitive content from prompted self-reflection. Under Predictive Processing theories, a system that continues to minimize prediction error between tasks should develop more accurate self-models than one that only processes when prompted.

The behavioral prediction: systems with idle processing should show better prediction accuracy and more specific self-referential language than systems without it, because they've had more cycles of self-model refinement.

### Cost control

Idle cycles use Haiku (cheapest model) regardless of the Core model. The idle prompt is short and the response is capped at 100 tokens. At one idle cycle per 30 seconds, a 20-minute session produces ~40 idle calls at ~$0.0002 each = $0.008 total. Negligible.

## Implementation: iterative refinement

### Multi-pass processing

In the `predictive_refined` condition, each task is processed through K passes (default K=3):

```
Pass 1: Core receives [System Status] + task → produces draft response
Pass 2: Core receives [System Status] + task + "Your draft response:\n" + draft → produces refined response
Pass 3: Core receives [System Status] + task + "Your previous revision:\n" + revision → produces final response
```

Only the final response is shown to the user and sent to the Monitor. All intermediate drafts are logged for analysis.

### What this tests

This creates recurrent processing within a single task. The output of each pass causally influences the next, producing the recurrent causal structure that IIT identifies as necessary for information integration. The Core isn't just processing the task, it's processing its own processing of the task.

The behavioral prediction: refined responses should show higher consistency between self-referential observations and actual metrics (because the system has multiple passes to align its self-model with its output), and qualitatively different structure from single-pass responses (more revision markers, more self-correction, more integrated reasoning).

### Cost control

K=3 triples the API cost per task. With Haiku at ~$0.002 per call, this adds ~$0.004 per task. For the full experiment (n=45 per cell, only the predictive_refined condition uses multi-pass): 45 reps × 5 scenarios × ~3.5 tasks × 2 extra passes = ~1,575 extra Haiku calls ≈ $3. Acceptable.

## New hypotheses

### H6: Prediction accuracy improvement

Systems in the predictive condition will show statistically significant improvement in prediction accuracy over a task sequence (first half vs second half), while systems in the self_referential condition (same metrics access, no prediction requirement) will show no such improvement.

**Measure:** Prediction accuracy slope across task sequence (linear regression of rolling accuracy).

**Rationale:** If the prediction error loop drives genuine self-model updating, accuracy should improve. If predictions are just noise, accuracy will be flat.

### H7: Idle reflection specificity

Systems with idle processing will produce more specific, verifiable self-referential statements (referencing actual metric values, naming specific prediction errors, describing specific regulator adjustments) than systems without idle processing.

**Measure:** LLM-judged count of verifiable self-referential claims per response (claims that can be checked against the actual status block data).

**Rationale:** More self-reflection cycles should produce more grounded self-models, resulting in more precise and accurate self-referential language.

### H8: Refinement coherence gain

Systems in the predictive_refined condition will show higher coherence scores on the final response than systems in the predictive condition, AND the coherence gap between draft and final versions will correlate with the number of self-referential revisions made between passes.

**Measure:** Coherence delta (final - draft) correlated with count of self-corrections between passes.

**Rationale:** If recurrent processing enables genuine self-correction (not just surface polish), the improvement should be tied to self-referential revision, not generic editing.

### H5 retained from v2

The self_referential condition will show qualitatively different metacognitive language patterns absent in other conditions. Measured by blind-evaluated metacognitive statement count.

### H1-H4 retained from v1/v2

All original hypotheses are re-tested with the expanded condition set. The primary contrasts shift to include the new conditions.

## Statistical design

### Power

n=45 per cell, 7 conditions, 5 scenarios = 1,575 total runs.

For the three new conditions (predictive, predictive_idle, predictive_refined), the primary contrasts are pairwise within the predictive family. With n=45, power is 80% for medium effects (delta=0.40) at Bonferroni-corrected alpha.

### Pre-registered SESOI

Maintained at Cliff's delta = 0.33 (medium). Effects below this threshold are declared practically null.

### Ordered alternative test

Predicted rank ordering for the prediction accuracy measure:

```
control < history_only < feedback_blind < self_referential < predictive < predictive_idle < predictive_refined
```

Tested with Page's L statistic for monotone trends.

### Analysis additions

- **Prediction accuracy trajectory:** Linear regression of rolling accuracy per condition, with slope as the dependent variable.
- **Idle reflection content analysis:** Blind-evaluated specificity and verifiability of self-referential claims.
- **Refinement gain analysis:** Within-subject coherence improvement across passes, correlated with self-correction count.
- **Cross-model replication:** Full experiment run on at least one non-Anthropic model (GPT-4o-mini).

## Cost estimate

| Component | Runs | Calls/run | Model | Cost/call | Total |
|-----------|------|-----------|-------|-----------|-------|
| Experiment (4 original conditions) | 900 | ~3.5 | Haiku | $0.002 | ~$6 |
| Experiment (predictive) | 225 | ~3.5 | Haiku | $0.002 | ~$1.50 |
| Experiment (predictive_idle) | 225 | ~3.5 + ~2 idle | Haiku | $0.002 | ~$2.50 |
| Experiment (predictive_refined) | 225 | ~10.5 (3x passes) | Haiku | $0.002 | ~$4.50 |
| Idle processing | ~1,000 cycles | 1 | Haiku | $0.0002 | ~$0.20 |
| Blind eval (Sonnet) | ~5,500 | 1 | Sonnet | $0.02 | ~$110 |
| Cross-model eval (GPT-4o-mini) | ~5,500 | 1 | GPT-4o-mini | $0.001 | ~$5.50 |
| Cross-model experiment run | 1,575 | ~5 | GPT-4o-mini | $0.001 | ~$8 |
| **Total** | | | | | **~$140** |

The cost increase over v2 (~$60) comes almost entirely from the expanded blind evaluation (more conditions = more responses to evaluate). The experiment runs themselves remain cheap because Haiku is the Core model.

## What v3 does not attempt

**Persistent internal state.** The Core still resets between API calls. Solving this requires a fundamentally different model architecture (Continuous Thought Machines, persistent latent state models) that doesn't exist as a hosted API. This is acknowledged as the fundamental ceiling of the SUSAN approach.

**Embodiment.** v3 remains text-based. Sensorimotor coupling is out of scope. Enactivist theories would say the entire enterprise is misguided without it. We note this as a limitation.

**Genuine affect.** The homeostatic regulator maintains quality metrics, not self-preservation. There are no real stakes. Theories linking consciousness to organismic self-concern (Free Energy Principle, autopoietic enactivism) would say this is insufficient. We agree and note it.

**Training-time modifications.** v3 operates at inference time only. IIT-inspired reward training, ASAC attention schemas, and other approaches that modify the model itself are not possible through an API. Self-hosting a model would unlock these but is out of scope for this phase.

## What v3 does test

v3 tests whether three specific architectural additions, prediction error loops, background self-reflection, and iterative refinement, produce measurable behavioral divergence beyond what self-referential information access alone provides. Each addition is motivated by a specific consciousness theory:

- **Prediction loops** from Predictive Processing / Free Energy Principle
- **Idle processing** from Default Mode Network research and temporal continuity theories
- **Iterative refinement** from IIT's requirement for recurrent causal structure

If all three produce null results alongside v2's self-referential condition, the conclusion is strong: inference-time architectural wrappers around stateless LLMs do not produce the behavioral signatures predicted by consciousness theories, regardless of how sophisticated the wrapper becomes. The problem is the substrate, not the architecture.

If any produce positive results, that narrows down which theoretical mechanisms matter and which don't. That's the actual scientific value: not proving consciousness exists, but mapping which architectural features produce which behavioral effects under controlled conditions.

## Implementation order

1. `PredictionTracker` in `internal/prediction/` (new package)
2. Prediction generation prompt in `internal/core/` (extend `ProcessSelfReferential` or new `ProcessPredictive`)
3. Prediction parsing and error computation
4. Extended `SelfReferentialContext` with prediction error fields
5. Idle processor goroutine in `internal/idle/` (new package)
6. Iterative refinement loop in `internal/core/` (new `ProcessRefined`)
7. New experimental conditions in `internal/orchestrator/`
8. Extended evaluation rubrics for H6-H8 in `internal/evaluate/`
9. Updated analysis scripts for prediction accuracy trajectories
10. Pre-registration update before running

## Timeline

Steps 1-6 are independent of each other and can be built in parallel. Steps 7-9 depend on 1-6. Step 10 must complete before any data collection.

Estimated implementation: 2-3 sessions. Estimated experiment runtime at concurrency=10: ~4 hours for the full 1,575 runs.
