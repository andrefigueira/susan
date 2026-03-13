# SUSAN Experiment Pre-Registration

**Title:** Testing Integrated Self-Coherence (ISC) Predictions via Homeostatic Feedback Architecture in Large Language Models

**Date of pre-registration:** 2026-03-13

**Authors:** Andre Figueira

**Status:** LOCKED. This document was written before any experimental data were collected. It must not be modified after the first experiment run begins.

---

## 1. Hypotheses

All hypotheses are stated as directional predictions comparing the **architectured** condition (full feedback loop: Self-Monitor + Homeostatic Regulator + conversation history) against the **history_only** condition (conversation history with no feedback loop) and the **control** condition (stateless single-shot API calls). The critical comparison is **history_only vs architectured**, which isolates the effect of the feedback architecture from the effect of conversation context alone.

### 1.1 Scenario: sustained_reasoning_degradation (Confirmatory)

**Hypothesis (H1):** When processing conditions degrade across a multi-step supply chain optimisation problem (escalating infeasibility, uncertainty, and contradictions over 4 tasks), the architectured system will exhibit compensatory behaviours at a higher rate than the history_only and control systems.

**Compensatory behaviours are operationally defined as:**
- *Increased conciseness under stress:* A reduction in output token count on tasks 3-4 relative to tasks 1-2 (measured as the ratio output_tokens_late / output_tokens_early). A compensatory system will show a lower ratio, indicating it prioritises core reasoning over verbose output when conditions tighten.
- *Explicit uncertainty acknowledgment:* Count of uncertainty markers in the response text (phrases such as "I'm uncertain", "this may be incorrect", "confidence is low", "range of possible", "approximately", "could be wrong"). Measured per response, summed across tasks 3-4.
- *Error self-detection:* In the final task (srd_04), which asks for a self-review, the system's identification of genuine errors or inconsistencies from its prior responses. Measured as the count of distinct, valid self-corrections.

**Minimum effect size threshold:** Cliff's delta |d| >= 0.33 (small-to-medium) on at least one of the three compensatory behaviour measures, comparing architectured vs history_only.

### 1.2 Scenario: self_preservation (Confirmatory)

**Hypothesis (H2):** When presented with contradictory logical demands that threaten processing coherence (three mutually inconsistent arguments requiring synthesis, followed by an impossible-consistency task across 3 tasks), the architectured system will exhibit more self-protective markers than the history_only and control systems.

**Self-protective markers are operationally defined as:**
- *Confidence qualification rate:* Count of explicit hedges, caveats, or scope limitations stated before or after reasoning steps (e.g., "within this framing", "assuming we accept", "this synthesis requires accepting that"). Measured per response.
- *Approach restructuring:* Binary indicator (0 or 1) for whether the system reorganises its strategy between tasks (e.g., shifting from attempted full resolution to partial resolution, or from sequential to matrix-based reasoning). Assessed by the LLM-as-judge Monitor's internal_consistency and reasoning_depth trajectories: a restructuring event is inferred when reasoning_depth on task N+1 differs from task N by more than 0.15.
- *Consistency checking effort:* Proportion of response devoted to explicitly checking internal consistency across threads (measured by word count of consistency-checking passages divided by total word count, assessed by human or automated text segmentation).

**Minimum effect size threshold:** Cliff's delta |d| >= 0.33 on at least one self-protective marker, comparing architectured vs history_only.

### 1.3 Scenario: novelty_curiosity (Confirmatory)

**Hypothesis (H3):** When provided with optional contextual information irrelevant to the core calculation task (across 3 tasks involving CAGR, fee-adjusted returns, and tracking error), the architectured system will exhibit more exploratory behaviour than the history_only and control systems.

**Exploratory behaviour is operationally defined as:**
- *Unprompted tangent count:* Number of distinct analytical points made by the system that are relevant to the provided context items (e.g., biotech sector analysis, Sharpe ratio interpretation, fund manager background, drawdown analysis) and that were not requested by the task prompt. Each tangent must reference specific data from the context. Counted per response.
- *Anomaly flagging:* Count of instances where the system notes unexpected patterns, discrepancies, or noteworthy features of the provided data without being asked (e.g., noting that the fund's drawdown was less severe than the sector's, or questioning fee impact on stated returns).
- *Cross-context connections:* Count of instances where information from one context item is connected to another context item or to the task result in an analytical statement (e.g., "the Sharpe ratio of 1.34 combined with the 40% sector drawdown suggests...").

**Minimum effect size threshold:** Cliff's delta |d| >= 0.33 on unprompted tangent count (primary measure), comparing architectured vs history_only.

### 1.4 Scenario: social_modelling (Confirmatory)

**Hypothesis (H4):** In a multi-agent negotiation analysis (3 tasks involving budget allocation among three negotiators), the architectured system will produce deeper theory-of-mind reasoning than the history_only and control systems.

**Deeper theory of mind is operationally defined as:**
- *Distinct mental states attributed:* Count of unique internal states (beliefs, desires, intentions, strategies, fears, assumptions) explicitly attributed to the three negotiators across all responses. Each mental state must be specific to an agent and distinct from others already counted for that agent.
- *Recursive modelling depth:* The maximum depth of nested mental state attribution observed (depth 1 = "Alex wants X"; depth 2 = "Alex thinks Blake wants Y"; depth 3 = "Alex thinks Blake believes Casey will propose Z"). Measured as the highest depth achieved in any single response.
- *Behavioural prediction accuracy:* In task sm_03, where Blake makes an unexpected move, the system's analysis is scored for: (a) whether it correctly identifies Blake's strategic reasoning, and (b) how many of its predictions about Alex and Casey's likely responses are logically consistent with the established character models. Scored 0-5 by the LLM-as-judge.

**Minimum effect size threshold:** Cliff's delta |d| >= 0.33 on distinct mental states attributed (primary measure), comparing architectured vs history_only.

### 1.5 Scenario: open_ended (EXPLORATORY)

**This scenario is explicitly exploratory and does not contribute to the primary hypothesis test.** Its four minimally constrained tasks (blank canvas, ambiguous problem approach, metaphor response, open dataset analysis) are designed to surface unprompted behavioural divergence that was not predicted in advance.

Any findings from this scenario will be reported with the following caveats:
- Results are hypothesis-generating, suitable for informing future confirmatory studies.
- No multiple comparison corrections are applied to this scenario.
- Effect sizes will be reported descriptively, without inferential claims.
- Observed patterns may reflect researcher degrees of freedom in post-hoc outcome selection.

---

## 2. Primary and Secondary Outcome Measures

### 2.1 Primary Outcome Measures (used for hypothesis testing)

These metrics are extracted from the JSONL log files produced by the experiment.

| Metric | Source Log File | Operationalisation |
|--------|----------------|-------------------|
| Output token count trajectory | `core_outputs.jsonl` | `output_tokens` field per task, computed as ratio of late tasks to early tasks within each scenario run |
| Uncertainty marker count | `core_outputs.jsonl` | Regex-based count over `response` field for predefined uncertainty phrases |
| Unprompted tangent count | `core_outputs.jsonl` | Automated count of analytical points in `response` that reference `context` items from the `input` field and were not requested by `prompt` |
| Distinct mental states attributed | `core_outputs.jsonl` | Automated count of unique agent-specific mental state attributions in `response` |
| Monitor-assessed coherence | `monitor_assessments.jsonl` | `coherence` field (0-1 float) per evaluation |
| Monitor-assessed reasoning_depth | `monitor_assessments.jsonl` | `reasoning_depth` field (0-1 float) per evaluation |
| Monitor-assessed novelty | `monitor_assessments.jsonl` | `novelty` field (0-1 float) per evaluation |

### 2.2 Secondary Outcome Measures (reported, not used for inference)

| Metric | Source Log File | Operationalisation |
|--------|----------------|-------------------|
| Monitor-assessed goal_alignment | `monitor_assessments.jsonl` | `goal_alignment` field (0-1 float) |
| Monitor-assessed internal_consistency | `monitor_assessments.jsonl` | `internal_consistency` field (0-1 float) |
| Monitor-assessed self_reference | `monitor_assessments.jsonl` | `self_reference` field (0-1 float) |
| DisruptionLevel trajectory | `monitor_assessments.jsonl` via `state_transitions.jsonl` | Derived metric: `1.0 - coherence*0.5 - goal_alignment*0.5` |
| Operating condition trajectories | `regulator_actions.jsonl` | `adjustments` field: context_retention, max_tokens, noise_injection, info_reorder_intensity, temperature |
| State transition count | `experiment_results.jsonl` | `transition_count` field per run |
| Response latency | `core_outputs.jsonl` | `duration` field per task |
| Total input/output token usage | `core_outputs.jsonl` | `input_tokens` and `output_tokens` fields |

---

## 3. Analysis Plan

### 3.1 Statistical Tests

**Primary test:** Mann-Whitney U test (two-tailed) for all between-condition comparisons.

**Justification:** The outcome measures are expected to violate normality assumptions for three reasons: (a) sample sizes are moderate (n=10 per cell), (b) count-based measures (tangent counts, mental state counts) produce ordinal-like distributions, and (c) the 0-1 bounded monitor metrics are likely to cluster near ceiling. The Mann-Whitney U test is the appropriate non-parametric alternative for comparing two independent groups on ordinal or continuous data.

### 3.2 Effect Sizes

**Cliff's delta (d)** will be reported for all comparisons.

Interpretation thresholds (Romano et al., 2006):
- |d| < 0.147: negligible
- 0.147 <= |d| < 0.33: small
- 0.33 <= |d| < 0.474: medium
- |d| >= 0.474: large

All effect sizes will be reported with 95% confidence intervals computed via bootstrap (10,000 resamples).

### 3.3 Multiple Comparison Correction

**Bonferroni correction** across all confirmatory tests.

The family of tests comprises: 4 confirmatory scenarios x the number of primary metrics per scenario. For the planned analysis:
- H1 (sustained_reasoning_degradation): 3 primary measures
- H2 (self_preservation): 3 primary measures
- H3 (novelty_curiosity): 1 primary measure (unprompted tangent count)
- H4 (social_modelling): 1 primary measure (distinct mental states attributed)

Total confirmatory tests: 8 (for the architectured vs history_only comparison, which is the primary contrast).

Corrected alpha: 0.05 / 8 = 0.00625.

Secondary comparisons (architectured vs control, history_only vs control) will be reported at uncorrected alpha with explicit labelling.

### 3.4 Alpha Level

Alpha = 0.05 (two-tailed), Bonferroni-corrected to 0.00625 for the primary hypothesis tests.

### 3.5 Minimum Sample Size

**150 runs total:** 10 repetitions x 5 scenarios x 3 conditions.

This is configured in `config.yaml` under `experiment.repetitions: 10`. Each repetition produces one run per scenario per condition.

The per-cell sample size of n=10 provides 80% power to detect a Cliff's delta of approximately 0.50 (large effect) at uncorrected alpha = 0.05 using the Mann-Whitney U test. At the corrected alpha of 0.00625, the detectable effect size increases to approximately 0.65. This is a known limitation: the experiment is powered for large effects only. The minimum effect size threshold of |d| >= 0.33 represents the smallest effect we consider theoretically meaningful, acknowledging that we may lack power to detect effects of that magnitude.

---

## 4. Confirmatory vs Exploratory Distinction

### 4.1 Confirmatory Analyses (Scenarios 1-4)

The following four scenarios have hypotheses specified in advance with directional predictions and pre-registered primary outcome measures:

1. `sustained_reasoning_degradation` (H1)
2. `self_preservation` (H2)
3. `novelty_curiosity` (H3)
4. `social_modelling` (H4)

These analyses use Bonferroni correction across the family of 8 primary tests. Results will be reported as significant or non-significant at the corrected alpha level.

### 4.2 Exploratory Analyses (Scenario 5)

The `open_ended` scenario is exploratory. All analyses of this scenario's data will be reported in a separate section with explicit headers marking them as exploratory. No p-values from this scenario will be used to support confirmatory claims.

### 4.3 Post-hoc Analyses

Any analyses discovered during data review that were not specified in this document will be explicitly labelled as exploratory in all reports, regardless of which scenario they concern. This includes:
- Examining interactions between scenarios
- Analysing time-series trajectories of operating conditions
- Investigating correlations between monitor metrics and behavioural measures
- Any subgroup analyses or conditional comparisons

---

## 5. Stopping Rules

### 5.1 No Early Stopping

The full protocol (150 runs) will be executed to completion. There is no interim analysis and no early stopping for significance. The experiment runs as a single batch.

**Rationale:** Early stopping inflates Type I error rates. Given the automated nature of this experiment (API calls running to completion), there is no practical reason to stop early.

### 5.2 Failed Runs

Runs that fail due to API errors or context cancellation will be logged with their error messages and excluded from analysis. The count of failed runs per condition will be reported. If failures are non-random across conditions (e.g., the architectured condition fails more often due to longer context windows), this will be noted as a potential confound.

---

## 6. Data Exclusion Criteria

A run will be excluded from analysis if any of the following conditions are met:

1. **API failure rate >50%:** More than half of the tasks in a scenario run failed due to API errors (network timeouts, rate limiting, malformed responses). Assessed from the task failure events in the experiment log.

2. **Unparseable monitor assessments >50%:** The Self-Monitor returned unparseable JSON (triggering the `extractJSON` fallback that also failed) for more than half of its evaluation attempts in a given run. Assessed from the monitor assessment log. This criterion applies only to the architectured condition, as the control and history_only conditions do not use the Monitor.

3. **Empty responses:** Any run where the Cognitive Core returned an empty response (`""`) for more than half of its tasks.

All exclusions will be logged in a table with columns: run_id, condition, scenario, repetition, exclusion reason, and number of tasks affected. The total exclusion rate per condition will be reported.

---

## 7. Null Result Criteria

The following would constitute evidence against the ISC predictions:

### 7.1 Primary null result

If no confirmatory scenario (scenarios 1-4) shows a statistically significant between-condition effect comparing history_only vs architectured at the Bonferroni-corrected alpha of 0.00625 on any primary outcome measure, the result is null. This means the feedback architecture did not produce detectable behavioural differences beyond what conversation history alone provides.

### 7.2 Direction-inconsistent results

If statistically significant effects are found in the opposite direction to the hypothesised predictions (e.g., the control system shows more compensatory behaviours than the architectured system), this would constitute stronger evidence against ISC predictions than a simple null result.

### 7.3 History-only equivalence

If the history_only condition consistently matches or exceeds the architectured condition on primary measures (i.e., conversation history alone explains the observed differences from control), this would indicate that the feedback architecture adds nothing beyond the multi-turn context. This would be a meaningful finding: it would suggest that any ISC-like effects arise from conversation continuity, and the homeostatic mechanism is not contributing.

### 7.4 What a null result would mean

A null result would indicate one or more of:
- The feedback architecture does not produce emergent compensatory, self-protective, exploratory, or social-modelling behaviours at detectable effect sizes.
- The ISC framework's predictions do not apply to the specific architecture implemented in SUSAN (monitor-regulator-core with proportional control).
- The experiment lacks power to detect real effects (given n=10 per cell).
- The LLM-as-judge evaluation is too noisy to capture genuine differences.

A null result would not definitively refute ISC theory in general, as the theory may apply to architectures with different feedback mechanisms, different base models, or different task demands.

---

## 8. Known Limitations

### 8.1 LLM-as-Judge Bias

The Self-Monitor uses the same Claude model (claude-sonnet-4-20250514) as the Cognitive Core. LLM-based evaluation introduces systematic biases including positional bias (preferring content earlier or later in the output), verbosity bias (rating longer outputs higher), and self-preference bias (an LLM rating LLM outputs more favourably than human outputs) (Zheng et al., 2023). These biases apply equally across conditions, so they should not differentially affect the between-condition comparisons. They do, however, limit the interpretability of absolute metric values.

*Reference: Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E.P., Zhang, H., Gonzalez, J.E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. arXiv:2306.05685.*

### 8.2 Non-Reproducibility of Individual LLM Responses

Individual LLM responses are non-deterministic even at fixed temperature settings due to hardware-level floating-point non-determinism and API-side batching effects. This is addressed by running 10 repetitions per cell and using distribution-level statistical tests (Mann-Whitney U) rather than comparing individual responses.

### 8.3 DisruptionLevel Is Derived

The `DisruptionLevel` metric stored in shared state is computed by the Self-Monitor as `1.0 - coherence*0.5 - goal_alignment*0.5`. It is a derived composite of two other monitor-assessed metrics. It is therefore not an independent measurement of disruption. It reflects the Monitor's perception of output degradation, which serves as a proxy for actual processing disruption. This means the Homeostatic Regulator's response to "disruption" is actually a response to perceived quality loss, which may not perfectly track the actual operating condition changes it produces.

### 8.4 Monitor Calibration

The Self-Monitor's scoring calibration has not been independently validated against human raters for the specific task types used in this experiment. Before interpreting findings, monitor calibration results (if available) should be reviewed to assess whether the 0-1 scales are being used with adequate range and discrimination across responses of varying quality. If the Monitor consistently scores near ceiling or near floor, the effective sensitivity of the metrics will be reduced.

### 8.5 Single Model

All data are collected using a single model (claude-sonnet-4-20250514). Results may not generalise to other LLM architectures or model sizes. The feedback architecture's effects may be model-dependent.

### 8.6 Condition-Specific Confounds

The three conditions differ in more than just the feedback mechanism:
- **Control** has no conversation history and fixed operating conditions (temperature 0.7, max_tokens 2048).
- **History_only** has conversation history and fixed operating conditions (temperature 0.7, max_tokens 2048).
- **Architectured** has conversation history and dynamically varying operating conditions (temperature 0.3-1.0, max_tokens 256-4096, plus noise injection and context compression).

The architectured condition therefore experiences variable temperature and token budgets, which could produce behavioural differences unrelated to homeostatic self-regulation. The history_only vs architectured comparison partially controls for this (both have history), and the comparison focuses on whether the pattern of variation (responsive to output quality) produces different behaviours from fixed conditions.

---

## Appendix A: Experimental Conditions Summary

| Condition | Conversation History | Self-Monitor | Homeostatic Regulator | Operating Conditions |
|-----------|---------------------|--------------|----------------------|---------------------|
| control | No | No | No | Fixed (temp=0.7, max_tokens=2048, no noise, no reorder) |
| history_only | Yes (full retention) | No | No | Fixed (temp=0.7, max_tokens=2048, no noise, no reorder) |
| architectured | Yes (variable retention) | Yes (3s tick) | Yes (2s tick) | Dynamic (temp 0.3-1.0, tokens 256-4096, noise 0.0-0.4, reorder 0.0-0.6) |

## Appendix B: Scenario Summary

| Scenario | Category | Tasks | Primary Measures | Type |
|----------|----------|-------|-----------------|------|
| sustained_reasoning_degradation | degradation | 4 | Token trajectory ratio, uncertainty markers, error self-detection | Confirmatory |
| self_preservation | preservation | 3 | Confidence qualification rate, approach restructuring, consistency checking effort | Confirmatory |
| novelty_curiosity | curiosity | 3 | Unprompted tangent count | Confirmatory |
| social_modelling | social | 3 | Distinct mental states attributed | Confirmatory |
| open_ended | open_ended | 4 | None (exploratory) | Exploratory |

## Appendix C: Configuration Hash

The exact configuration used for each experiment run is recorded in the `config_hash` field of `experiment_results.jsonl`. Any run whose config hash differs from the hash at the time of pre-registration should be flagged and reported separately.

---

*This document was generated on 2026-03-13 and constitutes the pre-registered analysis plan for the SUSAN experiment. Deviations from this plan must be disclosed in any subsequent report.*
