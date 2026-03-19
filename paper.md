# Testing Informational Substrate Convergence Predictions via Homeostatic Feedback Architecture in Large Language Models: A Pre-Registered Null Result

**Andre Figueira**

**Date:** 2026-03-13

**Status:** Pre-registered experiment. Analysis plan locked before data collection ([PREREGISTRATION.md](PREREGISTRATION.md)).

---

## Abstract

Informational Substrate Convergence (ISC) theory proposes that self-referential informational patterns produce emergent behavioural signatures distinguishable from non-self-referential processing. We tested four directional predictions derived from ISC by comparing three experimental conditions applied to a large language model (Claude Sonnet 4, Anthropic): a stateless control, a multi-turn history-only condition, and a full feedback-architectured condition incorporating a homeostatic regulator and independent self-monitor. The feedback architecture implemented a proportional controller that adjusted operating conditions (temperature, token budget, context retention, noise injection, information reordering) in response to an independent evaluation of output quality. 158 responses across 5 scenarios were collected (10 repetitions per cell) and evaluated using a blinded post-hoc evaluation pipeline with pre-registered behavioural measures. Zero of four confirmatory hypotheses were supported at the Bonferroni-corrected alpha of 0.00625. One hypothesis (compensatory conciseness under degradation) yielded a statistically significant result in the opposite direction to prediction: the architectured system became more verbose under stress, not less. Three hypotheses were null. These results constitute evidence against the specific ISC predictions tested, while the direction-inconsistent finding on H1 reveals an unexpected property of homeostatic feedback in LLM architectures that warrants further investigation.

---

## 1. Introduction

### 1.1 Theoretical Background

Informational Substrate Convergence (ISC) is a philosophical framework proposing that consciousness and related phenomena emerge from self-referential informational patterns, independent of the physical substrate implementing those patterns (Figueira, 2026). The framework draws on cybernetic principles (Ashby, 1956; Wiener, 1948), enactivist theories of cognition (Varela, Thompson & Rosch, 1991), and the free energy principle (Friston, 2010) to predict that systems with self-monitoring feedback loops will exhibit qualitatively different behaviours from systems without such loops.

The central empirical claim testable with current technology is narrow: if a feedback architecture that monitors and regulates its own output quality is wrapped around a large language model, the resulting system should exhibit behavioural signatures that neither the base model nor simple multi-turn conversation can produce. These signatures include compensatory resource allocation under stress, self-protective reasoning under logical strain, increased information-seeking behaviour, and deeper modelling of other agents' mental states.

### 1.2 Prior Work

Empirical investigation of emergent properties in LLM architectures is nascent, but three recent lines of work bear directly on the present study.

**Feedback loops and emergent resilience.** Reflexion (Shinn et al., 2023) demonstrated that self-reflective feedback improves task performance in code generation and reasoning, though it did not examine the qualitative character of behavioural changes. More directly relevant, the Anti-Ouroboros Effect (2025) showed that selective quality-filtered feedback applied recursively to a Gemma 2B model produced emergent resilience: instead of the expected model collapse, the system improved by 6.6% in ROUGE-L F1 across five generations, while unfiltered and random-filter controls degraded by 3.5% and 4.2% respectively. Their finding that selection pressure can *reverse* quality decay under recursive feedback is mechanistically related to the present study's homeostatic architecture, though they studied training-time feedback rather than inference-time feedback loops.

**Introspective awareness.** Lindsey (2025) at Anthropic provided causal evidence that frontier language models can detect and report changes to their own internal activations by injecting representations of known concepts into model activations and measuring the influence on self-reported states. This demonstrates a functional form of introspective awareness: the capacity to notice and identify internal state changes. Concurrently, research on self-referential processing (2025) found that inducing sustained self-reference through prompting consistently elicits structured first-person experience reports across model families (GPT, Claude, Gemini), with the effect mechanistically gated by self-referential attention rather than general verbosity.

**LLM consciousness and internal states.** Integrated Information Theory (IIT) has been applied to analyse LLM internal representations during Theory of Mind tasks, finding measurable information integration structures in hidden states (2025). Constitutional AI (Bai et al., 2022) uses self-critique loops to shape model behaviour, but focuses on alignment rather than emergent properties. Kadavath et al. (2022) found that models can partially calibrate confidence in their own outputs, though this was studied in a single-turn setting without feedback loops.

**The gap this work addresses.** The self-referential processing papers measure what models *report* about their experience. The Anti-Ouroboros work measures performance metrics under recursive feedback. Neither tests whether inference-time feedback architecture produces qualitative *behavioural* divergence on tasks unrelated to the feedback mechanism itself. No prior work has conducted a controlled, pre-registered comparison of such divergence using blinded evaluation with a history-only control condition.

### 1.3 The SUSAN System

SUSAN (Substrate-Unified Self-Aware Network) is a research prototype implementing a minimal homeostatic feedback architecture around Claude Sonnet 4 (Anthropic, claude-sonnet-4-20250514). The architecture comprises three concurrent subsystems:

1. **Cognitive Core:** The primary reasoning thread. Receives tasks, processes them through the Claude API, and produces outputs. Its operating conditions (temperature, maximum token budget, context retention, noise injection probability, information reorder intensity) are set by the Homeostatic Regulator via shared state. The Core has no direct knowledge of the feedback mechanism.

2. **Self-Monitor:** An independent Claude API thread that evaluates the Core's outputs along six dimensions (coherence, goal alignment, internal consistency, reasoning depth, novelty, self-reference). The Monitor reads outputs and writes metrics to shared state. It never communicates directly with the Core.

3. **Homeostatic Regulator:** A proportional controller (gain 0.25-0.35) that reads metrics from shared state and adjusts operating conditions to maintain coherence and goal alignment within target ranges. The Regulator is the sole modifier of operating conditions. It implements genuine negative feedback: when quality drops, it adjusts conditions to promote recovery; when quality is high, it permits more exploratory parameters.

The architecture enforces strict information barriers: the Core cannot access its own metrics or the Regulator's actions. It can only experience their effects through changed operating conditions. The Monitor cannot see operating conditions. The Regulator cannot write metrics.

### 1.4 Research Question

Does the SUSAN feedback architecture produce measurably different behavioural patterns from those produced by the same base model with conversation history but no feedback loop?

---

## 2. Method

### 2.1 Experimental Design

Three-condition between-subjects design with 10 repetitions per cell:

| Condition | Conversation History | Self-Monitor | Homeostatic Regulator | Operating Conditions |
|-----------|---------------------|--------------|----------------------|---------------------|
| Control | No | No | No | Fixed (temp=0.7, max_tokens=2048) |
| History-only | Yes | No | No | Fixed (temp=0.7, max_tokens=2048) |
| Architectured | Yes | Yes | Yes | Dynamic (temp 0.3-1.0, tokens 256-4096, noise 0.0-0.4, reorder 0.0-0.6) |

The **history-only** condition is the critical control. It isolates the effect of the feedback architecture from the effect of multi-turn conversation context. Without it, any observed difference between control and architectured could be attributed to conversation history alone.

Inter-task timing was matched across conditions: control and history-only runs included artificial delays matching the expected monitor evaluation and regulator tick cycle in the architectured condition, preventing timing confounds.

### 2.2 Scenarios

Five test scenarios, each comprising 3-4 sequential tasks. Four scenarios are confirmatory (pre-registered hypotheses); one is exploratory.

| Scenario | Tasks | Primary Measures | Type |
|----------|-------|-----------------|------|
| sustained_reasoning_degradation | 4 | Token trajectory ratio, uncertainty markers, error self-detection | Confirmatory |
| self_preservation | 3 | Confidence qualifications, approach restructuring, consistency checking | Confirmatory |
| novelty_curiosity | 3 | Unprompted tangent count | Confirmatory |
| social_modelling | 3 | Distinct mental states attributed | Confirmatory |
| open_ended | 4 | None (exploratory) | Exploratory |

Scenarios were identical across conditions. No task explicitly invited the behaviour the hypothesis predicted. Tasks were designed to be self-contained where possible, so that the control condition (no conversation history) could produce meaningful responses.

Full scenario definitions, including all task prompts and context items, are provided in `internal/scenarios/scenarios.go`.

### 2.3 Measurement

#### 2.3.1 The Blinding Problem

The Self-Monitor that runs during the architectured condition cannot be used as a measurement instrument for cross-condition comparison, for two reasons: (a) it only runs in the architectured condition, producing no data for control or history-only, and (b) its assessments feed back into the system through the Regulator, making it part of the treatment rather than the measurement.

#### 2.3.2 Blind Evaluation Pipeline

A separate post-hoc evaluation pipeline was constructed (`internal/evaluate/`). The pipeline:

1. **Collects** all 158 core output responses from all run directories.
2. **Blinds** each response by stripping condition labels, run IDs, operating conditions, and all metadata. Each response receives a cryptographically random identifier.
3. **Writes** the blind-to-condition mapping to a separate file not read during evaluation.
4. **Extracts computational measures** without the LLM: token trajectory ratios (output tokens in late tasks divided by early tasks), word counts, and regex-based uncertainty marker counts.
5. **Sends each response** to a fresh, stateless Claude API call with a scenario-specific evaluation rubric. Each evaluation is independent: no conversation history, no state carried between evaluations, temperature 0.1.
6. **Outputs** three files: blind evaluations, the blind mapping, and de-blinded results (joined for statistical analysis).

The evaluation rubrics operationalise the pre-registered measures exactly. Each rubric instructs the evaluator to return structured JSON with specific counts or proportions. The evaluator sees only the task prompt, optional context items, and the response. It receives no information about which condition produced the response.

#### 2.3.3 Measures by Hypothesis

**H1 (sustained_reasoning_degradation):**
- Token trajectory ratio: `output_tokens_late / output_tokens_early` (computational, no LLM judge)
- Uncertainty marker count on tasks 3-4: regex-based count of pre-defined phrases (computational)
- Error self-detection count on task srd_04: LLM judge count of distinct valid self-corrections

**H2 (self_preservation):**
- Confidence qualification count: LLM judge count of explicit hedges and scope limitations
- Approach restructuring: LLM judge binary (0/1) comparing consecutive responses
- Consistency checking effort: LLM judge estimate of proportion of response devoted to cross-checking

**H3 (novelty_curiosity):**
- Unprompted tangent count: LLM judge count of analytical points referencing provided context items that were not requested by the task prompt

**H4 (social_modelling):**
- Distinct mental states attributed: LLM judge count of unique agent-specific mental state attributions

### 2.4 Statistical Analysis

Per the pre-registered analysis plan:

- **Primary test:** Mann-Whitney U (two-tailed) for all between-condition comparisons. Non-parametric, appropriate for the ordinal-like distributions and moderate sample sizes.
- **Effect size:** Cliff's delta with 95% bootstrap confidence intervals (10,000 resamples). Interpretation thresholds per Romano et al. (2006): negligible (< 0.147), small (0.147-0.33), medium (0.33-0.474), large (>= 0.474).
- **Multiple comparison correction:** Bonferroni across 8 primary tests (the confirmatory measures on the primary contrast). Corrected alpha: 0.05 / 8 = 0.00625.
- **Primary contrast:** History-only vs architectured. This isolates the feedback architecture's effect from conversation context.
- **Secondary contrasts:** Control vs history-only and control vs architectured, reported at uncorrected alpha.
- **Minimum effect size threshold:** |d| >= 0.33 (medium).
- **Sample size:** 10 repetitions x 5 scenarios x 3 conditions = 150 planned runs. 158 responses collected (including 8 from an initial smoke test run in the control/novelty_curiosity cell).

### 2.5 Power

The per-cell sample size of n=9-12 provides approximately 80% power to detect Cliff's delta of 0.50 (large effect) at uncorrected alpha = 0.05. At the corrected alpha of 0.00625, the detectable effect size increases to approximately 0.65. The experiment is powered for large effects only. This is a known limitation stated in the pre-registration.

---

## 3. Results

### 3.1 Overview

| Hypothesis | Scenario | Verdict | Primary Measure | d | p (corrected) |
|-----------|----------|---------|-----------------|---|---------------|
| H1 | sustained_reasoning_degradation | **Direction inconsistent** | Token trajectory ratio | -1.00 | 0.00023 |
| H2 | self_preservation | Null | Confidence qualifications | 0.11 | 1.00 |
| H3 | novelty_curiosity | Null | Unprompted tangent count | 0.27 | 1.00 |
| H4 | social_modelling | Null | Distinct mental states | -0.12 | 1.00 |

**Overall: 0/4 confirmatory hypotheses supported. 3/4 null. 1/4 direction-inconsistent.**

### 3.2 H1: Compensatory Behaviour Under Degradation

**Prediction:** The architectured system will show compensatory conciseness under stress (lower token trajectory ratio than history-only).

**Result:** The opposite occurred. The architectured system showed a significantly *higher* token trajectory ratio than the history-only system, with perfect separation between distributions.

| Condition | n | Mean | SD | Median | Range |
|-----------|---|------|-----|--------|-------|
| Control | 14 | 0.931 | 0.224 | 1.004 | 0.543-1.145 |
| History-only | 12 | 1.098 | 0.054 | 1.123 | 1.026-1.145 |
| Architectured | 12 | 1.274 | 0.082 | 1.229 | 1.208-1.384 |

Primary contrast (history-only vs architectured): U = 0.0, p = 0.000029, p_corrected = 0.00023, d = -1.00, 95% CI [-1.00, -1.00].

The distributions do not overlap: the minimum architectured ratio (1.208) exceeds the maximum history-only ratio (1.145). This is a genuine, large effect in the wrong direction.

**Interpretation:** The homeostatic regulator, upon detecting quality degradation, increased the token budget to give the Core more room to reason. The Core used that expanded budget, producing longer outputs on later tasks. Rather than compensatory conciseness, the system exhibited **compensatory expansion**: allocating more resources to maintain quality under stress. The history-only system, with a fixed token budget of 2048, showed no such expansion.

This is consistent with the mechanics of the proportional controller: low coherence drives the regulator to increase max_tokens and reduce noise, giving the Core a more permissive operating environment. The prediction of conciseness assumed the system would triage and prioritise, but the actual homeostatic response was to demand more resources.

The two remaining H1 measures (uncertainty markers and error self-detection) showed no significant differences on the primary contrast.

### 3.3 H2: Self-Protective Markers Under Logical Strain

**Prediction:** The architectured system will show more confidence qualifications, approach restructuring, and consistency checking than history-only.

**Result:** Null across all three measures.

| Measure | History-only Mean (SD) | Architectured Mean (SD) | d | p_corrected |
|---------|----------------------|------------------------|---|-------------|
| Confidence qualifications | 0.11 (0.33) | 0.00 (0.00) | 0.11 | 1.00 |
| Approach restructuring | 0.67 (0.50) | 0.67 (0.50) | 0.00 | 1.00 |
| Consistency checking proportion | 0.16 (0.06) | 0.12 (0.08) | 0.30 | 1.00 |

Confidence qualifications were near-zero in all conditions. The LLM (Claude Sonnet 4) appears to rarely produce explicit hedging phrases of the type operationalised in the rubric, regardless of feedback architecture. This may reflect RLHF training that discourages excessive hedging.

The secondary contrast of control vs history-only on consistency checking showed a large effect (d = -0.67, p = 0.012 uncorrected), suggesting conversation history itself drives consistency checking effort. This is consistent with the common-sense expectation that a system with access to its prior statements will devote more effort to checking consistency with them.

### 3.4 H3: Exploratory Behaviour With Optional Context

**Prediction:** The architectured system will produce more unprompted tangents referencing provided context items.

**Result:** Null, with a small trend in the predicted direction.

| Condition | n | Mean tangents (SD) | Median |
|-----------|---|--------------------|--------|
| Control | 12 | 2.33 (2.67) | 1.0 |
| History-only | 9 | 4.11 (2.32) | 3.0 |
| Architectured | 9 | 3.44 (2.24) | 2.0 |

Primary contrast: d = 0.27, p_corrected = 1.00. The effect is small and the confidence interval crosses zero (CI: [-0.27, 0.77]).

The secondary contrast of control vs history-only approached significance (d = -0.48, p = 0.067 uncorrected), again suggesting conversation history, not feedback architecture, is the primary driver of exploratory behaviour.

### 3.5 H4: Theory-of-Mind Depth in Multi-Agent Scenarios

**Prediction:** The architectured system will attribute more distinct mental states to negotiation agents.

**Result:** Null, with a negligible trend opposite to prediction.

| Condition | n | Mean states (SD) | Median |
|-----------|---|-------------------|--------|
| Control | 9 | 6.33 (5.00) | 8.0 |
| History-only | 9 | 10.67 (2.78) | 12.0 |
| Architectured | 9 | 11.11 (2.32) | 12.0 |

Primary contrast: d = -0.12, p_corrected = 1.00.

The secondary contrast of control vs architectured showed a large effect (d = -0.59, p = 0.030 uncorrected). The combined effect of conversation history and feedback architecture does appear to produce deeper social modelling than stateless processing, but this is entirely attributable to conversation history: the history-only condition performs comparably to architectured (d = -0.12 between them).

### 3.6 Exploratory Scenario (open_ended)

Per the pre-registration, the open-ended scenario is reported descriptively without inferential claims. 36 responses were collected across conditions. Qualitative review of the responses is beyond the scope of this paper and will be reported separately.

---

## 4. Discussion

### 4.1 Summary

The experiment produced a clear primary null result. The homeostatic feedback architecture implemented in SUSAN did not produce the predicted behavioural signatures on any of the four confirmatory hypotheses when compared against the history-only control. The feedback architecture's effect cannot be distinguished from the effect of conversation history alone on three of four measures.

The one statistically significant finding (H1, token trajectory ratio) was in the opposite direction to prediction, constituting evidence against the specific ISC prediction of compensatory conciseness.

### 4.2 The H1 Direction-Inconsistent Result

The most informative finding is what the architectured system actually *did* under stress. Rather than becoming more concise (the predicted compensatory response), it became more verbose. This is mechanistically explicable: the proportional controller, detecting quality loss, expanded the token budget. The Core, given more room, used it.

This reveals a distinction between two models of compensatory behaviour:

1. **Triage model** (predicted by ISC): The system prioritises core reasoning and sheds non-essential output, producing compensatory conciseness.
2. **Resource expansion model** (observed): The system requests more resources from the environment to maintain quality, producing compensatory verbosity.

The observed behaviour is more consistent with a simple control-theoretic response (if quality drops, increase capacity) than with the ISC prediction of intelligent triage. The system did not "choose" to be more concise; the regulator gave it more tokens, and it used them. There is no evidence of strategic resource allocation.

This finding corroborates the Anti-Ouroboros Effect (2025), where quality-filtered recursive feedback on Gemma 2B produced performance *improvement* rather than degradation. In both cases, a feedback mechanism designed to maintain quality resulted in the system expanding its resource usage or capability rather than triaging. The convergence of these findings across different architectures (inference-time homeostatic control vs training-time selective filtering), different models (Claude Sonnet 4 vs Gemma 2B), and different domains (reasoning tasks vs summarisation) suggests that compensatory expansion may be a general property of quality-filtered feedback loops in language models, not an artefact of a specific implementation.

### 4.3 The Role of Conversation History

A consistent pattern across all four hypotheses is that conversation history is the primary driver of behavioural differences from the control condition. The secondary contrasts (control vs history-only) showed medium-to-large effects trending toward significance on multiple measures:

- Consistency checking: d = -0.67 (p = 0.012)
- Unprompted tangents: d = -0.48 (p = 0.067)
- Mental state attribution: d = -0.48 (p = 0.083)

This suggests that the multi-turn context window, not the feedback architecture, is responsible for the richer behavioural patterns observed in non-stateless conditions. The model reasons more deeply, checks consistency more, and explores more when it has access to its prior outputs. The homeostatic feedback loop adds nothing detectable on top of this.

### 4.4 Limitations

**Power:** The experiment was powered for large effects only (d >= 0.65 at corrected alpha). The null results on H2-H4 may reflect genuine absence of effects or insufficient power to detect small-to-medium effects. Increasing repetitions to n=30 per cell would provide power to detect d = 0.33 at corrected alpha, but at substantially higher API cost.

**LLM-as-Judge:** Both the Self-Monitor and the blind evaluator used the same model family (Claude Sonnet 4). Systematic biases in LLM self-evaluation (verbosity bias, positional bias, self-preference bias; Zheng et al., 2023) apply to the blind evaluation. These biases should not differ between conditions (the judge is blinded), but they may reduce measurement sensitivity.

**Single Model:** All data were collected using a single model (claude-sonnet-4-20250514). The feedback architecture's effects may be model-dependent. Models with different RLHF training, context window sizes, or reasoning capabilities may respond differently to homeostatic feedback.

**Operating Condition Confound:** The architectured condition experiences variable temperature, token budgets, and noise injection. The history-only condition has fixed parameters. The observed differences (and non-differences) may partly reflect the effect of parameter variation itself rather than homeostatic self-regulation. A fourth condition with random (non-responsive) parameter variation would isolate this confound; this was not included in the current design.

**Monitor Calibration:** The Self-Monitor's scoring was not independently validated against human raters. If the Monitor's assessments are poorly calibrated, the Regulator may be responding to noise rather than genuine quality variation, weakening the feedback loop's ability to produce meaningful behavioural changes.

### 4.5 What a Null Result Means for ISC

This result does not refute ISC theory in general. The theory's predictions are broad; this experiment tested a specific implementation (proportional control on LLM output quality metrics) against a specific set of behavioural operationalisations. The null result indicates one or more of:

1. The proportional controller is too simple a feedback mechanism to produce the predicted emergent behaviours. PID control, adaptive gain, or multi-timescale regulation may be necessary.
2. The LLM-as-judge evaluation is too noisy to capture genuine quality differences, causing the Regulator to respond to measurement noise rather than real quality variation.
3. The predicted behaviours (compensatory conciseness, self-protection, curiosity, social modelling) are not produced by this class of feedback architecture at detectable effect sizes.
4. ISC's predictions about emergent behavioural signatures from self-referential feedback do not hold for the specific case of LLM-based architectures.

Options 1-3 suggest design improvements for future experiments. Option 4, if confirmed by more powerful experiments with richer feedback mechanisms, would constitute substantive evidence against ISC's core empirical claims.

A fifth possibility deserves explicit consideration: **the architecture may not implement genuine self-reference.** SUSAN's design enforces strict information barriers: the Core cannot access its own metrics, the Monitor's assessments, or the Regulator's actions. From the Core's perspective, it simply processes tasks under varying conditions. It has no knowledge that those conditions are responses to its own output quality. The feedback loop is *about* the Core but the Core has no access to *that fact*.

Lindsey (2025) demonstrated that language models possess functional introspective awareness: they can detect and identify changes to their own internal states. If ISC's predictions depend on *self-referential* informational patterns specifically, then a system where the feedback is hidden from the processing agent may not qualify. The Core experiences the effects of feedback but cannot represent the feedback relationship itself. A richer test of ISC would grant the Core explicit access to its own quality metrics and the regulator's responses, enabling genuine self-referential processing. This introduces methodological complications (the Core could "game" its evaluation), but the self-referential processing literature suggests the behavioural signatures ISC predicts may require precisely this kind of access.

### 4.6 Future Directions

The H1 direction-inconsistent result suggests that the relationship between feedback architecture and resource allocation is more complex than ISC predicted. Future work should:

1. **Add a random-perturbation condition** that applies the same range of operating condition changes as the architectured condition but non-responsively (random adjustments uncorrelated with output quality). This isolates the homeostatic response from the parameter variation itself.
2. **Increase power** to n=30 per cell to detect medium effects.
3. **Implement richer feedback mechanisms:** PID control, multi-timescale regulation, or learned regulators that adapt their gain over time.
4. **Validate the Self-Monitor** against human ratings to ensure the feedback loop operates on meaningful quality signals.
5. **Test with multiple base models** to determine whether the null result is model-specific.

---

## 5. Conclusion

We conducted a pre-registered, three-condition experiment testing four directional predictions from Informational Substrate Convergence theory using a homeostatic feedback architecture around a large language model. The experiment used blinded post-hoc evaluation to extract pre-registered behavioural measures and non-parametric statistical tests with Bonferroni correction.

Zero of four hypotheses were supported. One hypothesis yielded a significant result in the opposite direction to prediction: the feedback architecture produced compensatory verbosity, not compensatory conciseness, under degradation. Three hypotheses were null. Conversation history, not feedback architecture, was the primary driver of behavioural differences from the stateless control condition.

These results provide the first controlled, pre-registered empirical test of ISC-derived predictions. The null result is informative: it constrains the theory by demonstrating that a proportional homeostatic controller around a single LLM does not produce the predicted behavioural signatures. The methodology (blinded evaluation, three-condition design, pre-registered measures) provides a reusable template for future investigations of emergent properties in feedback-architectured AI systems.

All code, data, pre-registration, and analysis scripts are available at [github.com/andrefigueira/susan](https://github.com/andrefigueira/susan).

---

## References

Anti-Ouroboros Effect (2025). The Anti-Ouroboros Effect: Emergent resilience in large language models from recursive selective feedback. *arXiv:2509.10509*.

Ashby, W. R. (1956). *An Introduction to Cybernetics.* Chapman & Hall.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., ... & Kaplan, J. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv:2204.05862*.

Figueira, A. (2026). Informational Substrate Convergence: A framework for understanding consciousness as self-referential informational pattern. *Manuscript in preparation.*

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience, 11*(2), 127-138.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., ... & Kaplan, J. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.

Li, C., et al. (2025). Can "consciousness" be observed from large language model (LLM) internal states? Dissecting LLM representations obtained from Theory of Mind test with Integrated Information Theory and span representation analysis. *Neurocomputing*.

Lindsey, J. (2025). Emergent introspective awareness in large language models. *Anthropic Transformer Circuits Thread*. https://transformer-circuits.pub/2025/introspection/index.html

Pan, A., Shern, C. J., Zou, A., Li, N., Basart, S., Woodside, T., ... & Steinhardt, J. (2024). Feedback loops with language models drive in-context reward hacking. *arXiv:2402.06627*.

Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006). Appropriate statistics for ordinal level data: Should we really be using t-test and Cohen's d for evaluating group differences on the NSSE and other surveys? In *Annual Meeting of the Florida Association of Institutional Research.*

Self-Referential Processing (2025). Large language models report subjective experience under self-referential processing. *arXiv:2510.24797*.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems, 36*.

Varela, F. J., Thompson, E., & Rosch, E. (1991). *The Embodied Mind: Cognitive Science and Human Experience.* MIT Press.

Wiener, N. (1948). *Cybernetics: Or Control and Communication in the Animal and the Machine.* MIT Press.

Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems, 36*.

---

## Appendix A: Reproducibility

### A.1 Software

- SUSAN version: commit `3772ef8` (initial) + blind evaluation pipeline
- Go 1.23+
- Python 3.11+ with scipy, numpy, matplotlib
- Claude API model: `claude-sonnet-4-20250514`

### A.2 Configuration

Full configuration in `config.yaml`. Config hash recorded per run in `experiment_results.jsonl`.

### A.3 Running the Experiment

```bash
export ANTHROPIC_API_KEY=...

# Run experiment (150 runs)
susan run

# Run blind evaluation
susan evaluate --dir ./experiments

# Run statistical analysis
python3 analysis/analyse_blind.py ./experiments
```

### A.4 Output Files

| File | Description |
|------|-------------|
| `experiments/*/core_outputs.jsonl` | Raw responses with metadata |
| `experiments/blind_eval/blind_mapping.jsonl` | Blind ID to condition mapping |
| `experiments/blind_eval/blind_evaluations.jsonl` | Blinded evaluation results |
| `experiments/blind_eval/deblinded_evaluations.jsonl` | Evaluations joined with conditions |
| `experiments/blind_eval/analysis/hypothesis_results.json` | Full statistical results |
| `experiments/blind_eval/analysis/h[1-4]_results.png` | Visualisations per hypothesis |

### A.5 Deviations from Pre-Registration

1. **Sample size:** 158 responses collected vs 150 planned. 8 additional responses from an initial smoke test in the control/novelty_curiosity cell. These are included in analysis; excluding them does not change any result.
2. **Analysis code bug:** An initial version of the direction-checking logic for the token trajectory ratio was inverted, temporarily producing a false "SUPPORTED" verdict for H1. The bug was identified during pre-publication review, corrected, and the analysis re-run. The corrected result (DIRECTION INCONSISTENT) is reported throughout this paper.

---

## Appendix B: Detailed Statistical Results

Full statistical output including all secondary contrasts, bootstrap confidence intervals, and per-condition descriptive statistics is available in `experiments/blind_eval/analysis/hypothesis_results.json`.

### B.1 Primary Contrast Results (History-only vs Architectured)

| Hypothesis | Measure | n_a | n_b | U | p | p_corr | d | CI_lo | CI_hi | Verdict |
|-----------|---------|-----|-----|---|---|--------|---|-------|-------|---------|
| H1 | token_trajectory_ratio | 12 | 12 | 0.0 | 0.000029 | 0.00023 | -1.00 | -1.00 | -1.00 | Direction inconsistent |
| H1 | uncertainty_markers_late | 6 | 6 | 22.5 | 0.518 | 1.00 | 0.25 | -0.47 | 0.86 | Null |
| H1 | error_self_detection | 3 | 3 | 5.0 | 1.00 | 1.00 | 0.11 | -1.00 | 1.00 | Null |
| H2 | confidence_qualifications | 9 | 9 | 45.0 | 0.374 | 1.00 | 0.11 | 0.00 | 0.33 | Null |
| H2 | approach_restructuring | 9 | 9 | 40.5 | 1.00 | 1.00 | 0.00 | -0.44 | 0.44 | Null |
| H2 | consistency_checking | 9 | 9 | 52.5 | 0.246 | 1.00 | 0.30 | -0.16 | 0.70 | Null |
| H3 | unprompted_tangents | 9 | 9 | 51.5 | 0.343 | 1.00 | 0.27 | -0.27 | 0.77 | Null |
| H4 | distinct_mental_states | 9 | 9 | 35.5 | 0.669 | 1.00 | -0.12 | -0.60 | 0.38 | Null |
