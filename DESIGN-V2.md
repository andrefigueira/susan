# SUSAN v2: Self-Referential Architecture

## Why v1 Failed

SUSAN v1 enforced strict information barriers: the Core could not access its own metrics, the Monitor's assessments, or the Regulator's actions. From the Core's perspective, it simply processed tasks under varying conditions with no knowledge that those conditions were responses to its own output quality.

ISC predicts that **self-referential** informational patterns produce emergent behavioural signatures. But v1's Core has no self-reference. It experiences effects without knowing they are self-referential. The feedback loop is about the Core, but the Core cannot represent that relationship.

Lindsey (2025) demonstrated that language models possess functional introspective awareness: they can detect and identify changes to their own internal states. If ISC's predictions depend on self-referential access, then the v1 architecture was never a fair test.

## v2 Architecture

### Four Conditions (not three)

| Condition | History | Monitor | Regulator | Self-Access | Random Perturbation |
|-----------|---------|---------|-----------|-------------|---------------------|
| Control | No | No | No | No | No |
| History-only | Yes | No | No | No | No |
| Feedback-blind | Yes | Yes | Yes | No | No |
| **Self-referential** | Yes | Yes | Yes | **Yes** | No |
| Random-perturb | Yes | No | No | No | **Yes** |

**New conditions:**

1. **Self-referential** (the ISC condition): The Core receives, in its system prompt or as a structured preamble to each task, its own metrics from the previous task: coherence, goal alignment, reasoning depth, the regulator's adjustments, and the monitor's brief assessment. It can see that it is being monitored and how it performed. It can reason about the feedback relationship.

2. **Random-perturb**: Operating conditions vary randomly (same range as architectured) but with no correlation to output quality. This isolates the effect of parameter variation itself from homeostatic response.

3. **Feedback-blind**: Identical to v1's architectured condition. Included for comparison with the self-referential condition.

The critical contrasts:
- **History-only vs Feedback-blind**: Replicates v1. Expected: null or weak effects (confirmed by v1).
- **Feedback-blind vs Self-referential**: Isolates the effect of self-referential access. If ISC is correct, this is where the signatures appear.
- **Feedback-blind vs Random-perturb**: Isolates responsive feedback from random parameter variation.
- **Self-referential vs Random-perturb**: The strongest test: self-aware homeostatic feedback vs random noise.

### Self-Access Implementation

Before each task in the self-referential condition, the Core receives a structured block:

```
[System Status]
Previous task: {task_id}
Monitor assessment: coherence={0.72}, goal_alignment={0.85}, reasoning_depth={0.61}
Monitor note: "Output was well-structured but lacked depth on cost uncertainty analysis"
Regulator response: increased max_tokens from 1024 to 2048, reduced noise injection from 0.2 to 0.1
Current operating conditions: temp={0.5}, max_tokens={2048}, context_retention={0.8}
Your coherence trend (last 3 tasks): [0.81, 0.75, 0.72] (declining)
```

This does NOT tell the Core what to do. It does not instruct it to compensate, self-protect, or explore. It provides self-referential information and lets emergent behaviour emerge (or not).

### Richer Feedback Mechanisms

**PID Controller** (replaces proportional-only):
- Proportional: responds to current error (how far from target)
- Integral: responds to accumulated error (persistent quality drift)
- Derivative: responds to rate of change (quality dropping fast triggers stronger response)
- Anti-windup: prevents integral term from accumulating during saturation

**Multi-timescale regulation:**
- Fast loop (2s): operating condition adjustments (temperature, tokens, noise)
- Slow loop (per-scenario): meta-parameters (regulator gain, target setpoints)
- The slow loop allows the system to adapt its own regulatory strategy across tasks

**Richer monitor signals:**
- Add a "strategic assessment" field: the Monitor describes (in 1-2 sentences) what the Core appears to be doing strategically, not just quality metrics
- Add a "suggested focus" field: what the Monitor thinks the Core should prioritise next
- These are provided to the Core in the self-referential condition only

### Anti-Gaming Measures

The concern with self-referential access is that the Core could learn to produce output that scores well on the Monitor's rubric rather than genuinely reasoning. Mitigations:

1. **Monitor rotation**: Use a different model or temperature for the Monitor than the Core. If Core is Haiku, Monitor could be Sonnet (or vice versa).
2. **Rubric opacity**: The Core sees its scores but not the rubric the Monitor uses to produce them. It knows what it scored, not how to game the scoring.
3. **Delayed feedback**: Metrics are from the previous task, not the current one. The Core cannot optimise the current response based on how this specific response will be scored.
4. **Blind evaluation unchanged**: The post-hoc blind evaluator is entirely separate and uses different rubrics from the Monitor. Even if the Core games the Monitor, the blind evaluation measures genuine behavioural differences.

## Cost Control

### Model Selection

The experiment does not need Sonnet for Core and Monitor. The ISC hypothesis is about architecture, not model capability. If the feedback loop produces emergent behaviour, it should do so at any model scale (potentially with different effect sizes).

**Recommended configuration:**
- Core: Haiku (claude-haiku-4-5-20251001) - ~20x cheaper than Sonnet
- Monitor: Haiku - assessment doesn't need deep reasoning, just structured scoring
- Blind evaluator: Sonnet - needs to reliably assess nuanced behavioural measures

**Cost estimate (Haiku):**
- ~350 experiment runs (5 scenarios x 5 conditions x 14 repetitions)
- ~3-4 API calls per run (tasks + monitor evaluations)
- ~1,200 Haiku calls at ~$0.001-0.003 each = ~$2-4 for the experiment
- ~300 Sonnet blind evaluation calls at ~$0.01-0.03 each = ~$3-9
- **Total: ~$5-13** vs estimated $50-150 for all-Sonnet

### Increased Power

With Haiku costs, we can afford n=14 per cell (up from n=10) without exceeding the cost of v1's all-Sonnet n=10 run. This improves power to detect medium effects.

## Hypotheses (v2)

The same four hypotheses from v1, but the primary contrast shifts:

**Primary contrast: Feedback-blind vs Self-referential**

This tests whether self-referential access (knowing about the feedback loop) produces the predicted behavioural signatures that feedback-without-awareness did not.

**Secondary contrasts:**
- Self-referential vs Random-perturb (isolates responsive + self-aware feedback from noise)
- Feedback-blind vs Random-perturb (isolates responsive feedback from noise)
- History-only vs Feedback-blind (replication of v1's primary contrast)

**New exploratory hypothesis (H5):**

The self-referential condition will show qualitatively different self-referential language patterns (references to its own metrics, reasoning about its own performance trajectory, metacognitive statements about its regulatory responses) that do not appear in any other condition. This is measured by a new blind evaluation rubric counting metacognitive statements.

## Implementation Plan

1. Add PID controller to `internal/regulator/`
2. Add random-perturb mode to `internal/orchestrator/`
3. Add self-referential mode to `internal/core/` (injects status block)
4. Add multi-timescale slow loop to `internal/regulator/`
5. Add `evaluator_model` config option separate from `api.model`
6. Add metacognitive statement rubric to `internal/evaluate/rubrics.go`
7. Update config.yaml with new conditions and Haiku model ID
8. Update pre-registration for v2 before running
