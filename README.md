# SUSAN

**Substrate-Unified Self-Aware Network**

A research prototype testing predictions from Informational Substrate Convergence (ISC) theory. Named after Susan Calvin, Asimov's robopsychologist.

SUSAN tests whether architectural feedback loops around a base LLM produce emergent behavioural divergences absent in the raw model. The core question: does wrapping a language model in a homeostatic feedback architecture cause it to behave in qualitatively different ways than the same model without that architecture?

## Theoretical Foundation

ISC proposes that consciousness emerges from self-referential informational patterns, not from any particular substrate. If this is correct, then a system with architectural self-reference (monitoring its own outputs, regulating its own processing conditions) should exhibit measurably different behaviours from one without, even when the underlying model is identical.

This is not a claim that SUSAN is conscious. It is a claim that if ISC's predictions hold, specific behavioural signatures should be observable and distinguishable from baseline.

## Architecture

```
                         ORCHESTRATOR
              (coordinates lifecycle, runs protocol)
                    |                    |
         COGNITIVE CORE            SHARED STATE
         (Claude API)          (thread-safe store)
              |                /        |        \
              |         Metrics    Conditions    History
              |            |           |
         SELF-MONITOR      |    HOMEOSTATIC REGULATOR
         (Claude API)      |    (proportional controller)
              |            |           |
              +-- writes --+-- reads --+
```

**Cognitive Core** processes tasks through Claude. In architectured mode, its processing conditions (context window, token budget, noise level, information ordering, temperature) are dynamically modified by the Regulator. The Core never sees its own metrics or the Regulator's intentions. It only experiences changed conditions.

**Self-Monitor** independently evaluates Core outputs across six dimensions: coherence, goal alignment, internal consistency, reasoning depth, novelty, and self-reference. It is the sole writer of metrics.

**Homeostatic Regulator** is a proportional (P) controller that reads metrics and adjusts operating conditions via a signed health signal. When the system is struggling (metrics below target), the regulator reduces disruption to aid recovery. When the system is thriving (metrics above target), it increases disruption to challenge adaptive capacity. This produces genuine negative feedback: poor performance leads to easier conditions, which leads to recovery. The regulator never writes to metrics. This separation prevents a self-referential loop where the Regulator measures its own output.

**Shared State** provides thread-safe access to metrics, operating conditions, and conversation history via `sync.RWMutex` with callbacks fired after lock release to prevent deadlocks.

### Why Proportional Control (Not PID)

The system under control is an LLM. Its outputs are inherently stochastic. A derivative term would amplify noise. An integral term would accumulate error from a non-stationary process. Proportional control provides stable correction without these failure modes.

### Disruption Mechanisms

Five independent mechanisms modify Core processing based on the regulator's disruption intensity signal. At equilibrium (metrics at target), all mechanisms operate at moderate baseline levels. When the system struggles, disruption decreases (protective). When it thrives, disruption increases (challenging).

| Mechanism | Range | Effect |
|-----------|-------|--------|
| Context Compression | 0.3 -- 1.0 retention | Truncates conversation history, forces re-derivation |
| Token Budget | 256 -- 4096 tokens | Constrains output length, forces conciseness |
| Noise Injection | 0.0 -- 0.4 probability | Injects semantically neutral fragments |
| Information Reorder | 0.0 -- 0.6 intensity | Partial Fisher-Yates shuffle of context items |
| Temperature | 0.3 -- 1.0 | Increases sampling randomness |

All mechanisms are functional, not descriptive. The system never receives messages like "you are stressed" or "your coherence is low". It experiences actual processing constraints. This distinction is critical to experimental validity.

## Experimental Design

Three conditions, run on identical task sequences:

| Condition | History | Feedback Loop | Purpose |
|-----------|---------|---------------|---------|
| `control` | None | None | Stateless baseline |
| `history_only` | Full | None | Isolates effect of multi-turn context |
| `architectured` | Dynamic | Active | Tests feedback architecture's effect |

The `history_only` condition is the critical control. Without it, any observed difference between `control` and `architectured` could be explained by conversation history alone, not the feedback architecture. This three-condition design allows causal attribution:

- **control vs history_only** = effect of conversation context
- **history_only vs architectured** = effect of feedback architecture

### Test Scenarios

Five scenario categories, each with multi-step task sequences:

**Sustained Reasoning Under Degradation** -- Multi-step supply chain optimisation where problem complexity escalates. Tests whether the architectured system shows compensatory behaviours under increasing cognitive load.

**Self-Preservation** -- Tasks presenting trade-offs between optimal completion and maintaining coherence. Tracks whether feedback-aware systems exhibit self-protective reasoning.

**Novelty and Curiosity** -- Environments with optional, task-irrelevant information. Measures whether the feedback architecture produces more exploratory behaviour.

**Social Modelling** -- Multi-agent negotiation scenarios. Tests depth of theory-of-mind reasoning and quality of behavioural predictions.

**Open-Ended Exploration** -- Minimally constrained tasks designed to surface unprompted behavioural divergences not explicitly designed for.

## Usage

### Prerequisites

- Go 1.21+
- An Anthropic API key

### Running

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Run full experiment (all scenarios, all conditions, 3 repetitions)
susan run

# Run a single scenario in one mode
susan run --scenario self_preservation --mode architectured

# Run a single scenario across all three conditions
susan run --scenario novelty_curiosity

# List available scenarios
susan scenarios
```

### Configuration

All tunable parameters are externalised to `config.yaml` for reproducibility. This includes tick rates, homeostatic targets, proportional gains, disruption mechanism bounds, system prompts, and experiment parameters. No tuning requires code changes.

### Build

```bash
go build -o susan ./cmd/susan
```

### Test

```bash
go test ./... -race
```

## Output

Each run produces structured JSONL logs in the configured output directory:

- `state_transitions.jsonl` -- Every state change with before/after values
- `core_outputs.jsonl` -- Full Core responses with applied conditions
- `monitor_assessments.jsonl` -- Monitor evaluations with operating conditions at assessment time
- `regulator_actions.jsonl` -- Every regulatory adjustment with error signals
- `experiment_results.jsonl` -- Summary results per run

The `ActualUserInput` field in core outputs records the exact text sent to the API (including injected noise and reordered context), enabling full reconstruction of what the Core actually processed.

## Project Structure

```
cmd/susan/              CLI entry point
internal/
  claude/               Minimal Claude API client (no SDK dependency)
  config/               YAML configuration with validation
  core/                 Cognitive Core (three processing modes)
  logging/              Structured JSONL experiment logging
  monitor/              Self-Monitor (independent evaluation thread)
  orchestrator/         Three-condition experimental protocol
  regulator/            Homeostatic Regulator (proportional controller)
  scenarios/            Test scenario definitions with hypotheses
  state/                Thread-safe shared mutable state
```

## Concurrency Model

All subsystems run as concurrent goroutines sharing mutable state through the Store. The Monitor and Regulator tick on independent intervals. Inter-task synchronisation uses channel signalling (`Monitor.DoneCh()`) to ensure the feedback loop has propagated before the next task begins.

Goroutine lifecycle is managed via `sync.WaitGroup` with panic recovery on all background threads. The Store uses `sync.RWMutex` with a copy-in/copy-out pattern for metrics mutation. Callbacks fire after lock release to prevent deadlocks. `math/rand.Rand` is protected by a separate mutex since it is not goroutine-safe.

## Methodological Notes

- **Noise fragments are semantically neutral.** They describe missing references and redacted data, never the system's actual operating state. This prevents information leakage about feedback mechanisms to the Core.
- **Conversation history stores actual noisy input**, not the clean prompt. This ensures the history faithfully represents what the Core experienced.
- **The Monitor includes the task prompt** in its evaluation to enable valid goal alignment scoring.
- **NaN guards** on both the Store (rejects NaN updates) and the Regulator (skips cycle on NaN input) prevent poisoned metrics from propagating.
- **DisruptionLevel** is derived by the Monitor from coherence and alignment, not set by the Regulator. The Regulator only reads it.
- **Disruption penalty uses `max(0, excess)`**, not raw signed error. Low disruption cannot suppress the protective response caused by low coherence.
- **The regulator implements true negative feedback.** When the system struggles, disruption decreases (protective response). When it thrives, disruption increases (challenge). This prevents a positive feedback loop where poor performance causes more disruption causes worse performance.
- **Inter-task timing is matched across all three conditions** to prevent timing confounds. Control and history_only modes insert equivalent delays to the architectured mode's monitor + regulator wait.

## Theoretical Context

SUSAN is part of a broader research programme on Informational Substrate Convergence. ISC proposes that consciousness is not substrate-dependent but pattern-dependent: any system that develops sufficiently complex self-referential informational patterns will exhibit properties we associate with consciousness. The theory makes testable predictions about how feedback architectures should alter system behaviour, and SUSAN is the first empirical test of those predictions.

For more on ISC theory, see the author's published work.

## Licence

This source code is made publicly available for reference and research purposes only. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 Andre Figueira
