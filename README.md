# SUSAN

**Substrate-Unified Self-Aware Network**

A pre-registered experiment testing whether wrapping an LLM in a homeostatic feedback architecture produces emergent behavioural signatures predicted by Informational Substrate Convergence (ISC) theory. Named after Susan Calvin, Asimov's robopsychologist, the one who actually took robot cognition seriously as a scientific discipline while everyone else was scared of it or dismissing it.

## What this actually does

SUSAN takes a base LLM, wraps it in a real-time feedback loop with an independent self-monitor and a PID-based homeostatic regulator, then runs controlled experiments to see if the feedback architecture makes the model behave in qualitatively different ways. Five experimental conditions, five scenario categories, blinded post-hoc evaluation with pre-registered measures and Bonferroni-corrected statistics. Provider-agnostic, runs on Claude or any OpenAI-compatible API.

The v1 experiment ran. The results are in. **Zero of four hypotheses were supported.** One finding went in the opposite direction. The null result is published here alongside the code because negative results matter.

v2 addresses the critical flaw v1 exposed: the Core couldn't see its own metrics. It experienced the effects of feedback but couldn't represent the feedback relationship itself. v2 gives the Core explicit self-referential access, adds two new control conditions, cross-model evaluation to catch shared training biases, and a human rating pipeline for gold-standard validation.

## Try it: presence mode

The fastest way to see what SUSAN does is to talk to it directly.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
go build -o susan ./cmd/susan
./susan presence
```

Opens a web UI at `http://localhost:3000`. You chat with SUSAN while the full self-referential feedback loop runs live behind it:

- The **Self-Monitor** evaluates every response (coherence, alignment, depth, novelty, self-reference) on a 3-second tick
- The **Homeostatic Regulator** adjusts operating conditions (temperature, token budget, context retention) on a 2-second tick based on those evaluations
- The **self-referential injection** feeds SUSAN its own metrics, performance trends, and the regulator's response before each reply
- **Server-sent events** push state changes to the UI in real time so you can watch the feedback loop as it operates

SUSAN sees its own performance data. It can reason about why its coherence is declining, notice when the regulator has expanded its token budget, or observe that its responses are trending in a particular direction. Whether this produces something qualitatively different from a model without that self-referential access is exactly what the experiment tests.

The presence mode uses Sonnet for the Core (configurable in `config.yaml` under `presence.model`) and Haiku for the Monitor.

## Architecture

```
                         ORCHESTRATOR
              (coordinates lifecycle, runs protocol)
                    |                    |
         COGNITIVE CORE            SHARED STATE
         (LLM client)          (thread-safe store)
              |                /        |        \
              |         Metrics    Conditions    History
              |            |           |
         SELF-MONITOR      |    HOMEOSTATIC REGULATOR
         (LLM client)      |    (PID controller)
              |            |           |
              +-- writes --+-- reads --+
```

**Cognitive Core** processes tasks through an LLM (Claude, GPT, or any OpenAI-compatible provider). In feedback conditions, its operating conditions (temperature, token budget, context retention, noise injection, information reordering) are dynamically modified by the Regulator. In the self-referential condition, the Core also receives its own metrics, trend data, and the Regulator's actions before each task. This provides self-referential information at the prompt level (contextual self-reference), not mechanistic self-reference (direct access to weights or activations). The experiment tests whether this level of self-referential information is sufficient to produce the ISC-predicted behavioural signatures.

**Self-Monitor** independently evaluates Core outputs across six dimensions: coherence, goal alignment, internal consistency, reasoning depth, novelty, and self-reference. Sole writer of metrics.

**Homeostatic Regulator** reads metrics and adjusts operating conditions via a PID controller. When the system struggles, it reduces disruption to aid recovery. When it thrives, it increases disruption to challenge adaptive capacity. True negative feedback, not positive. The Regulator never writes metrics. The Monitor never reads conditions. Strict information barriers.

**Shared State** provides thread-safe access via `sync.RWMutex` with copy-in/copy-out mutation and callbacks fired after lock release.

## Experimental conditions

| Condition | History | Monitor | Regulator | Self-access | Purpose |
|-----------|---------|---------|-----------|-------------|---------|
| `control` | No | No | No | No | Stateless baseline |
| `history_only` | Yes | No | No | No | Isolates conversation context |
| `feedback_blind` | Yes | Yes | Yes | No | Feedback loop, Core can't see metrics |
| `self_referential` | Yes | Yes | Yes | Yes | Core sees own metrics and regulatory actions |
| `random_perturb` | Yes | No | No | No | Random parameter variation, no feedback correlation |

The critical contrasts:

- **history_only vs feedback_blind** isolates the feedback architecture from conversation context
- **feedback_blind vs self_referential** isolates self-referential access (the ISC prediction)
- **feedback_blind vs random_perturb** isolates responsive feedback from random noise
- **self_referential vs random_perturb** the strongest test: self-aware homeostasis vs noise

## Disruption mechanisms

Five independent mechanisms modify Core processing. All are functional, not descriptive. The system never receives messages like "you are stressed." It experiences actual processing constraints.

| Mechanism | Range | Effect |
|-----------|-------|--------|
| Context compression | 0.3 - 1.0 retention | Truncates history, forces re-derivation |
| Token budget | 256 - 4096 tokens | Constrains output length |
| Noise injection | 0.0 - 0.4 probability | Injects semantically neutral fragments |
| Information reorder | 0.0 - 0.6 intensity | Partial Fisher-Yates shuffle of context |
| Temperature | 0.3 - 1.0 | Sampling randomness |

## v1 results

350 runs across 5 conditions and 5 scenarios. 158 responses evaluated through a blinded pipeline (evaluator never saw condition labels). Pre-registered measures tested with Mann-Whitney U, Bonferroni-corrected at alpha = 0.00625.

| Hypothesis | Prediction | Result | Effect size (Cliff's delta) | p-value |
|------------|-----------|--------|----------------------------|---------|
| H1: Compensatory conciseness | Architectured becomes concise under stress | **Opposite direction**: became MORE verbose | d = -1.00 | 0.00023 |
| H2: Self-protective reasoning | More confidence qualifications | Null | d = 0.11 | 0.68 |
| H3: Exploratory behaviour | More unprompted tangents | Null (small trend in predicted direction) | d = 0.27 | 0.31 |
| H4: Theory of mind depth | More mental state attributions | Null | d = -0.12 | 0.71 |

The H1 reversal is the interesting finding. The regulator expanded the token budget when quality dropped (protective response), and the Core used every token it was given. Compensatory expansion, not compensatory conciseness. The system behaved like an organism given more resources under stress and using all of them, which is arguably more biologically plausible than our original prediction of triage-like conciseness.

Secondary analysis showed conversation history (not the feedback architecture) is the primary driver of behavioural divergence between conditions.

**Full paper:** [paper.md](paper.md)

## v2: self-referential access

v1's critical limitation: the Core experienced the effects of feedback but couldn't represent the feedback relationship itself. This is like adjusting someone's thermostat without telling them, they feel the temperature change but can't reason about why.

v2 injects a structured status block before each task in the self-referential condition:

```
[System Status]
Previous task: srd_02
Monitor assessment: coherence=0.72, goal_alignment=0.85, reasoning_depth=0.61
Monitor note: "Well-structured but lacked depth on cost uncertainty analysis"
Regulator response: increased max_tokens 1024 → 2048, reduced noise 0.2 → 0.1
Your coherence trend (last 3 tasks): [0.81, 0.75, 0.72] (declining)
```

No instructions on what to do with this information. Just self-referential data. Let behaviour emerge or not.

Anti-gaming measures: rubric opacity (Core sees scores, not how they're computed), delayed feedback (previous task metrics, not current), separate blind evaluator with different rubrics, monitor model rotation.

**New in v2:**

- **Self-referential condition** with explicit metric injection
- **Random-perturb condition** isolating parameter noise from feedback
- **PID controller** replacing proportional-only (integral for persistent drift, derivative for rate-of-change, anti-windup clamping)
- **Cross-model evaluation** to detect shared training biases when Claude evaluates Claude
- **Human rating pipeline** for gold-standard validation independent of any LLM
- **Multi-provider support** to test ISC predictions across model families
- **n=45 per cell** (up from n=14), powered for medium effects at corrected alpha
- **Pre-registered SESOI** (smallest effect size of interest) and ordered-alternative tests

**Full v2 design:** [DESIGN-V2.md](DESIGN-V2.md)

## Running the experiment

```bash
# Prerequisites: Go 1.23+, an API key
export ANTHROPIC_API_KEY=sk-ant-...
go build -o susan ./cmd/susan

# Full experiment (5 conditions x 5 scenarios x 45 reps = 1,125 runs, ~$30 with Haiku)
./susan run

# Single scenario in one condition
./susan run --scenario self_preservation --mode self_referential

# Single scenario across all five conditions
./susan run --scenario novelty_curiosity

# List available scenarios and their hypotheses
./susan scenarios
```

### Running on a different model

SUSAN is provider-agnostic. To run on GPT-4o-mini or any OpenAI-compatible API, change `config.yaml`:

```yaml
api:
  key: "${OPENAI_API_KEY}"
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
  provider: "openai"
```

Or keep a separate config file and pass it with `--config config-gpt.yaml`.

## Evaluation

### Blind LLM evaluation

The evaluator strips condition labels, assigns cryptographic blind IDs, and scores responses using pre-registered rubrics. The evaluator never sees which condition produced which response.

```bash
# Run blind evaluation with the primary model (Sonnet by default)
./susan evaluate --dir ./experiments

# Results land in:
#   experiments/blind_eval/blind_evaluations.jsonl    (blinded scores)
#   experiments/blind_eval/blind_mapping.jsonl         (ID-to-condition map)
#   experiments/blind_eval/deblinded_evaluations.jsonl (joined for analysis)
```

### Cross-model evaluation

To check whether Claude's evaluation biases affect results, enable a second evaluator from a different model family. Both evaluators score every response independently, and the analysis reports inter-rater agreement.

```yaml
# In config.yaml:
cross_evaluation:
  enabled: true
  provider: "openai"
  key: "${OPENAI_API_KEY}"
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
```

Cross-model results are stored in `cross_model_judgments` alongside the primary `llm_judgments` in the evaluation output.

### Human rating

For gold-standard evaluation independent of any LLM's training biases:

```bash
# Export blinded responses as CSV with blank rubric columns
./susan export-for-rating --dir ./experiments --output human_ratings.csv

# Distribute the CSV to human raters. They fill in the blank columns.
# Import completed ratings back into the pipeline:
./susan import-ratings --csv human_ratings.csv --output human_evaluations.jsonl
```

The CSV contains blind IDs, task prompts, and responses. No condition labels. Human raters score each response on the same rubrics the LLM evaluator uses.

### Statistical analysis

```bash
# After evaluation, run the analysis pipeline
python3 analysis/analyse.py ./experiments
```

The analysis computes Mann-Whitney U tests for pairwise contrasts, Cliff's delta effect sizes, Bonferroni correction, and Page's L trend test across the predicted condition ordering. Effects smaller than the pre-registered SESOI (Cliff's delta = 0.33) are declared practically null regardless of statistical significance.

## Output

Each run produces structured JSONL:

```
experiments/
  {condition}_{scenario}_rep{n}_{timestamp}/
    core_outputs.jsonl          # Full responses with applied conditions
    monitor_assessments.jsonl   # Independent quality evaluations
    regulator_actions.jsonl     # Every regulatory adjustment with error signals
    state_transitions.jsonl     # Operating condition snapshots
    experiment_results.jsonl    # Summary per run
  blind_eval/
    blind_mapping.jsonl         # Condition-blinded ID mapping
    blind_evaluations.jsonl     # Evaluator scores (blinded)
    deblinded_evaluations.jsonl # Scores joined with conditions
```

`ActualUserInput` in core outputs records the exact text sent to the API (including injected noise, reordered context), enabling full reconstruction of what the Core actually processed.

## Configuration

All tunable parameters are externalised to `config.yaml`. No tuning requires code changes. Config hash recorded with each run for reproducibility.

Key sections:

- **`api`** - Provider, model, API key, evaluator model
- **`cross_evaluation`** - Second evaluator for bias detection
- **`homeostasis`** - PID gains, targets, and anti-windup limits per metric
- **`disruption`** - Mechanism ranges (context compression, token budget, noise, reorder, temperature)
- **`experiment`** - Repetitions, seed, concurrency, power analysis parameters
- **`analysis`** - Pre-registered SESOI, trend test, predicted condition ordering
- **`presence`** - Interactive mode port, model, system prompt

## Project structure

```
cmd/susan/              CLI entry point (run, evaluate, presence, export/import ratings)
internal/
  llm/                  Provider-agnostic LLM client interface
    anthropic.go        Anthropic Messages API implementation
    openai.go           OpenAI Chat Completions API implementation
  config/               YAML config with validation
  core/                 Cognitive Core (five processing modes + self-referential injection)
  evaluate/             Blinded evaluation pipeline + human rating export/import
  logging/              Structured JSONL experiment logging
  monitor/              Self-Monitor (independent evaluation thread)
  orchestrator/         Five-condition experimental protocol
  presence/             Interactive web UI with live feedback loop
  regulator/            Homeostatic Regulator (PID controller)
  scenarios/            Test scenario definitions with hypotheses
  state/                Thread-safe shared mutable state
analysis/
  analyse.py            Statistical analysis (Mann-Whitney U, effect sizes, Page's L)
  analyse_blind.py      Blinded analysis pipeline
  calibration.py        Monitor calibration checks
```

## Concurrency model

All subsystems run as concurrent goroutines sharing mutable state through the Store. The Monitor and Regulator tick on independent intervals (3s and 2s respectively). Inter-task synchronisation uses channel signalling to ensure the feedback loop propagates before the next task begins.

Goroutine lifecycle managed via `sync.WaitGroup` with panic recovery. Store uses `sync.RWMutex` with copy-in/copy-out mutation. NaN guards on both Store and Regulator prevent poisoned metrics from propagating. `math/rand.Rand` protected by separate mutex.

## Methodological notes

- Noise fragments are semantically neutral (missing references, redacted data), never information about the system's actual state
- Conversation history stores noisy input, not clean prompts, so history faithfully represents what the Core experienced
- Monitor includes the task prompt to enable valid goal alignment scoring
- Inter-task timing matched across all conditions to prevent timing confounds
- The regulator implements true negative feedback: struggling reduces disruption, thriving increases it
- Disruption penalty uses `max(0, excess)`, low disruption cannot suppress protective response from low coherence
- Self-referential access is prompt-level contextual self-reference, not mechanistic introspection. The experiment tests whether this is sufficient for ISC predictions
- Cross-model evaluation addresses the shared training bias concern when an LLM evaluates its own family
- n=45 per cell provides 80% power to detect medium effects (Cliff's delta = 0.40) at Bonferroni-corrected alpha = 0.00625
- Effects below the pre-registered SESOI (delta = 0.33) are declared practically null
- v2 cost estimate: ~$30 for Haiku Core/Monitor, ~$22 for Sonnet evaluation, ~$22 for cross-model evaluation. Total ~$60-75

## Licence

Source code made publicly available for reference and research purposes only. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 Andre Figueira
