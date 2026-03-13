#!/usr/bin/env python3
"""
SUSAN Experiment Analysis Pipeline.

Performs statistical analysis of the three-condition ISC experiment:
  - Control: stateless single-shot API calls
  - History-only: multi-turn with conversation history, no feedback
  - Architectured: multi-turn with conversation history AND feedback loop

Statistical methodology:
  - Non-parametric tests (Mann-Whitney U) because LLM outputs are not
    normally distributed and sample sizes may be small.
  - Cliff's delta for effect sizes (does not assume normality or equal
    variance, appropriate for ordinal/non-normal data).
  - Bonferroni correction for multiple comparisons.
  - Bootstrap confidence intervals on Cliff's delta.

The history-only condition is the critical control that isolates the
feedback architecture's effect from the effect of mere conversation context.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITIONS = ["control", "history_only", "architectured"]
CONDITION_LABELS = {
    "control": "Control",
    "history_only": "History-only",
    "architectured": "Architectured",
}
METRIC_DIMENSIONS = [
    "coherence",
    "goal_alignment",
    "internal_consistency",
    "reasoning_depth",
    "novelty",
    "self_reference",
    "disruption_level",
]
CONDITION_PAIRS = [
    ("control", "history_only"),
    ("history_only", "architectured"),
    ("control", "architectured"),
]

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CI = 0.95


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping malformed lines."""
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  warning: skipping malformed JSON at {path}:{line_no}", file=sys.stderr)
    except OSError as e:
        print(f"  warning: could not read {path}: {e}", file=sys.stderr)
    return entries


def discover_run_dirs(data_dir: Path) -> list[Path]:
    """Find all run directories (any directory containing JSONL files)."""
    run_dirs = set()
    for jsonl_file in data_dir.rglob("*.jsonl"):
        run_dirs.add(jsonl_file.parent)
    return sorted(run_dirs)


def load_all_data(data_dir: Path) -> dict:
    """
    Load all experiment data from run directories.

    Returns a dict with keys:
      experiment_results, core_outputs, monitor_assessments,
      regulator_actions, state_transitions, state_snapshots, task_failures
    Each value is a list of unwrapped entry dicts.
    """
    file_map = {
        "experiment_results": "experiment_results.jsonl",
        "core_outputs": "core_outputs.jsonl",
        "monitor_assessments": "monitor_assessments.jsonl",
        "regulator_actions": "regulator_actions.jsonl",
        "state_transitions": "state_transitions.jsonl",
        "state_snapshots": "state_snapshots.jsonl",
        "task_failures": "task_failures.jsonl",
    }

    all_data = {k: [] for k in file_map}
    run_dirs = discover_run_dirs(data_dir)

    if not run_dirs:
        print(f"error: no JSONL files found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(run_dirs)} run directories under {data_dir}")

    for run_dir in run_dirs:
        for key, filename in file_map.items():
            fpath = run_dir / filename
            if fpath.exists():
                entries = load_jsonl(fpath)
                all_data[key].extend(entries)

    for key, entries in all_data.items():
        print(f"  {key}: {len(entries)} entries")

    return all_data


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def extract_experiment_results(raw: list[dict]) -> list[dict]:
    """Extract experiment result data, unwrapping the envelope."""
    results = []
    for entry in raw:
        data = entry.get("data", entry)
        if isinstance(data, dict) and "mode" in data:
            results.append(data)
    return results


def extract_monitor_assessments(raw: list[dict]) -> list[dict]:
    """Extract monitor assessment data."""
    assessments = []
    for entry in raw:
        data = entry.get("data", entry)
        if isinstance(data, dict) and "coherence" in data:
            a = dict(data)
            a["_run_id"] = entry.get("run_id", "")
            a["_seq"] = entry.get("seq", 0)
            a["_timestamp"] = entry.get("timestamp", "")
            assessments.append(a)
    return assessments


def extract_regulator_actions(raw: list[dict]) -> list[dict]:
    """Extract regulator action data."""
    actions = []
    for entry in raw:
        data = entry.get("data", entry)
        if isinstance(data, dict) and "health" in data:
            a = dict(data)
            a["_run_id"] = entry.get("run_id", "")
            a["_seq"] = entry.get("seq", 0)
            a["_timestamp"] = entry.get("timestamp", "")
            actions.append(a)
    return actions


def extract_state_transitions(raw: list[dict]) -> list[dict]:
    """Extract state transition data."""
    transitions = []
    for entry in raw:
        data = entry.get("data", entry)
        if isinstance(data, dict) and "field" in data:
            t = dict(data)
            t["_run_id"] = entry.get("run_id", "")
            transitions.append(t)
    return transitions


def extract_core_outputs(raw: list[dict]) -> list[dict]:
    """Extract core output data."""
    outputs = []
    for entry in raw:
        data = entry.get("data", entry)
        if isinstance(data, dict) and "response" in data:
            o = dict(data)
            o["_run_id"] = entry.get("run_id", "")
            o["_seq"] = entry.get("seq", 0)
            outputs.append(o)
    return outputs


def get_mode_from_run_id(run_id: str) -> str:
    """Infer condition mode from a run_id string."""
    for mode in CONDITIONS:
        if run_id.startswith(mode):
            return mode
    return "unknown"


# ---------------------------------------------------------------------------
# Per-task metric extraction from experiment results
# ---------------------------------------------------------------------------

def collect_per_task_metrics(results: list[dict]) -> dict:
    """
    Build per-condition, per-scenario metric arrays from experiment results.

    For control and history_only runs, metrics come from task_outputs
    (there is no monitor, so we do not have per-task assessments from
    the monitor). For architectured runs, we also have monitor assessments.

    Returns: {condition: {scenario: {metric: [values]}}}
    """
    metrics_by_cond = {c: defaultdict(lambda: defaultdict(list)) for c in CONDITIONS}
    return metrics_by_cond


def collect_assessment_metrics(assessments: list[dict]) -> dict:
    """
    Build per-condition metric arrays from monitor assessments.

    Monitor assessments only exist for architectured runs, but we group
    by run_id prefix to be safe.

    Returns: {condition: {metric: [values]}}
    """
    metrics = {c: defaultdict(list) for c in CONDITIONS}
    for a in assessments:
        run_id = a.get("_run_id", "")
        mode = get_mode_from_run_id(run_id)
        if mode not in metrics:
            continue
        for dim in METRIC_DIMENSIONS:
            val = a.get(dim)
            if val is not None and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                metrics[mode][dim].append(float(val))
    return metrics


def collect_final_metrics(results: list[dict]) -> dict:
    """
    Collect final_metrics from experiment_results for each condition/scenario.

    Returns: {condition: {scenario: {metric: [values]}}}
    """
    data = {c: defaultdict(lambda: defaultdict(list)) for c in CONDITIONS}
    for r in results:
        mode = r.get("mode", "")
        scenario = r.get("scenario", "unknown")
        fm = r.get("final_metrics", {})
        if not fm or mode not in data:
            continue
        for dim in METRIC_DIMENSIONS:
            val = fm.get(dim)
            if val is not None:
                data[mode][scenario][dim].append(float(val))
    return data


def collect_per_task_from_outputs(core_outputs: list[dict]) -> dict:
    """
    Collect per-task behavioural data from core_outputs.

    Returns: {condition: {"response_lengths": [], "input_tokens": [],
              "output_tokens": [], "durations": []}}
    """
    data = {c: defaultdict(list) for c in CONDITIONS}
    for o in core_outputs:
        mode = o.get("mode", "")
        if not mode:
            mode = get_mode_from_run_id(o.get("_run_id", ""))
        if mode not in data:
            continue
        resp = o.get("response", "")
        data[mode]["response_lengths"].append(len(resp))
        data[mode]["input_tokens"].append(o.get("input_tokens", 0))
        data[mode]["output_tokens"].append(o.get("output_tokens", 0))
        dur = o.get("duration")
        if dur is not None:
            # Duration may be a Go duration (nanoseconds as int) or a float.
            if isinstance(dur, (int, float)):
                data[mode]["durations"].append(float(dur) / 1e9)  # ns to seconds
    return data


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def cliffs_delta(x, y):
    """
    Compute Cliff's delta, a non-parametric effect size measure.

    Cliff's delta = (number of x>y pairs - number of x<y pairs) / (n_x * n_y)

    Range: [-1, 1]
    Interpretation (Romano et al., 2006):
      |d| < 0.147: negligible
      |d| < 0.33:  small
      |d| < 0.474: medium
      |d| >= 0.474: large
    """
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return float("nan"), "undefined"

    more = 0
    less = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                more += 1
            elif xi < yi:
                less += 1

    delta = (more - less) / (n_x * n_y)

    abs_d = abs(delta)
    if abs_d < 0.147:
        interpretation = "negligible"
    elif abs_d < 0.33:
        interpretation = "small"
    elif abs_d < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return delta, interpretation


def cliffs_delta_bootstrap_ci(x, y, n_boot=BOOTSTRAP_ITERATIONS, ci=BOOTSTRAP_CI):
    """
    Bootstrap confidence interval for Cliff's delta.

    Resamples both x and y with replacement and computes Cliff's delta
    on each bootstrap sample. Returns the percentile-based CI.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed=42)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        d, _ = cliffs_delta(bx.tolist(), by.tolist())
        deltas[i] = d

    alpha = (1 - ci) / 2
    lo = float(np.nanpercentile(deltas, 100 * alpha))
    hi = float(np.nanpercentile(deltas, 100 * (1 - alpha)))
    return (lo, hi)


def run_pairwise_tests(metrics_by_condition: dict, scenarios: list[str]) -> list[dict]:
    """
    Run Mann-Whitney U tests between each pair of conditions for each metric.

    Applies Bonferroni correction across all tests (metrics x pairs x scenarios).

    Returns a list of result dicts.
    """
    results = []

    # First pass: collect all tests to determine Bonferroni denominator.
    test_specs = []
    for metric in METRIC_DIMENSIONS:
        for cond_a, cond_b in CONDITION_PAIRS:
            # Pooled across scenarios.
            test_specs.append(("pooled", metric, cond_a, cond_b))
            # Per-scenario.
            for scenario in scenarios:
                test_specs.append((scenario, metric, cond_a, cond_b))

    n_tests = len(test_specs)
    if n_tests == 0:
        return results

    # Second pass: run tests.
    for scope, metric, cond_a, cond_b in test_specs:
        if scope == "pooled":
            x = metrics_by_condition.get(cond_a, {}).get(metric, [])
            y = metrics_by_condition.get(cond_b, {}).get(metric, [])
        else:
            # For per-scenario, we need scenario-specific data. We will
            # handle this with the final_metrics data passed separately.
            continue  # handled below

        if len(x) < 2 or len(y) < 2:
            continue

        try:
            u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            continue

        corrected_p = min(p_value * n_tests, 1.0)
        delta, interpretation = cliffs_delta(x, y)
        ci_lo, ci_hi = cliffs_delta_bootstrap_ci(x, y)

        results.append({
            "scope": scope,
            "metric": metric,
            "condition_a": cond_a,
            "condition_b": cond_b,
            "n_a": len(x),
            "n_b": len(y),
            "U_statistic": float(u_stat),
            "p_value": float(p_value),
            "p_corrected": float(corrected_p),
            "bonferroni_n": n_tests,
            "cliffs_delta": float(delta),
            "effect_interpretation": interpretation,
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "significant_005": corrected_p < 0.05,
            "significant_001": corrected_p < 0.01,
        })

    return results


def run_scenario_tests(final_metrics: dict, n_pooled_tests: int) -> list[dict]:
    """
    Run per-scenario Mann-Whitney U tests with Bonferroni correction.

    The Bonferroni denominator includes BOTH pooled and per-scenario tests
    to properly control family-wise error rate.
    """
    results = []

    # Count all scenario tests to add to Bonferroni denominator.
    all_scenarios = set()
    for cond in CONDITIONS:
        if cond in final_metrics:
            all_scenarios.update(final_metrics[cond].keys())
    scenarios = sorted(all_scenarios)

    scenario_test_count = 0
    for metric in METRIC_DIMENSIONS:
        for cond_a, cond_b in CONDITION_PAIRS:
            for scenario in scenarios:
                x = final_metrics.get(cond_a, {}).get(scenario, {}).get(metric, [])
                y = final_metrics.get(cond_b, {}).get(scenario, {}).get(metric, [])
                if len(x) >= 2 and len(y) >= 2:
                    scenario_test_count += 1

    n_total = n_pooled_tests + scenario_test_count

    for metric in METRIC_DIMENSIONS:
        for cond_a, cond_b in CONDITION_PAIRS:
            for scenario in scenarios:
                x = final_metrics.get(cond_a, {}).get(scenario, {}).get(metric, [])
                y = final_metrics.get(cond_b, {}).get(scenario, {}).get(metric, [])
                if len(x) < 2 or len(y) < 2:
                    continue
                try:
                    u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
                except ValueError:
                    continue

                corrected_p = min(p_value * n_total, 1.0)
                delta, interpretation = cliffs_delta(x, y)
                ci_lo, ci_hi = cliffs_delta_bootstrap_ci(x, y)

                results.append({
                    "scope": scenario,
                    "metric": metric,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "n_a": len(x),
                    "n_b": len(y),
                    "U_statistic": float(u_stat),
                    "p_value": float(p_value),
                    "p_corrected": float(corrected_p),
                    "bonferroni_n": n_total,
                    "cliffs_delta": float(delta),
                    "effect_interpretation": interpretation,
                    "ci_lower": float(ci_lo),
                    "ci_upper": float(ci_hi),
                    "significant_005": corrected_p < 0.05,
                    "significant_001": corrected_p < 0.01,
                })

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(values: list[float]) -> dict:
    """Compute descriptive statistics for a list of values."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "iqr": None,
                "q1": None, "q3": None, "min": None, "max": None}
    arr = np.array(values, dtype=float)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return {
        "n": len(values),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "iqr": q3 - q1,
        "q1": q1,
        "q3": q3,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def build_summary_table(metrics_by_condition: dict) -> dict:
    """
    Build per-condition summary statistics for each metric dimension.

    Returns: {condition: {metric: {stat_name: value}}}
    """
    summary = {}
    for cond in CONDITIONS:
        summary[cond] = {}
        cond_data = metrics_by_condition.get(cond, {})
        for metric in METRIC_DIMENSIONS:
            values = cond_data.get(metric, [])
            summary[cond][metric] = compute_summary_stats(values)
    return summary


def build_scenario_summary(final_metrics: dict) -> dict:
    """
    Build per-condition, per-scenario summary statistics.

    Returns: {condition: {scenario: {metric: {stat_name: value}}}}
    """
    summary = {}
    for cond in CONDITIONS:
        summary[cond] = {}
        if cond not in final_metrics:
            continue
        for scenario in sorted(final_metrics[cond].keys()):
            summary[cond][scenario] = {}
            for metric in METRIC_DIMENSIONS:
                values = final_metrics[cond][scenario].get(metric, [])
                summary[cond][scenario][metric] = compute_summary_stats(values)
    return summary


# ---------------------------------------------------------------------------
# Behavioural divergence
# ---------------------------------------------------------------------------

def behavioural_analysis(core_outputs: list[dict], assessments: list[dict],
                         state_transitions: list[dict], results: list[dict]) -> dict:
    """Compute behavioural divergence metrics."""
    output_data = collect_per_task_from_outputs(core_outputs)

    analysis = {"response_lengths": {}, "token_usage": {},
                "self_reference": {}, "novelty": {},
                "transition_counts": {}, "operating_condition_trajectories": {}}

    # Response length distributions.
    for cond in CONDITIONS:
        lengths = output_data.get(cond, {}).get("response_lengths", [])
        analysis["response_lengths"][cond] = compute_summary_stats(lengths)

    # Token usage.
    for cond in CONDITIONS:
        analysis["token_usage"][cond] = {
            "input_tokens": compute_summary_stats(
                output_data.get(cond, {}).get("input_tokens", [])),
            "output_tokens": compute_summary_stats(
                output_data.get(cond, {}).get("output_tokens", [])),
        }

    # Self-reference and novelty from monitor assessments.
    assess_metrics = collect_assessment_metrics(assessments)
    for cond in CONDITIONS:
        analysis["self_reference"][cond] = compute_summary_stats(
            assess_metrics.get(cond, {}).get("self_reference", []))
        analysis["novelty"][cond] = compute_summary_stats(
            assess_metrics.get(cond, {}).get("novelty", []))

    # State transition counts (architectured only).
    arch_transitions = [t for t in state_transitions
                        if get_mode_from_run_id(t.get("_run_id", "")) == "architectured"]
    field_counts = defaultdict(int)
    for t in arch_transitions:
        field_counts[t.get("field", "unknown")] += 1
    analysis["transition_counts"] = dict(field_counts)
    analysis["total_transitions_architectured"] = len(arch_transitions)

    # Transition counts per run from experiment results.
    for r in results:
        if r.get("mode") == "architectured":
            run_id = r.get("run_id", "")
            tc = r.get("transition_count", 0)
            analysis.setdefault("transition_counts_per_run", {})[run_id] = tc

    # Operating condition trajectories (architectured, from core outputs).
    arch_outputs = [o for o in core_outputs if
                    o.get("mode", "") == "architectured" or
                    get_mode_from_run_id(o.get("_run_id", "")) == "architectured"]
    arch_outputs.sort(key=lambda o: o.get("_seq", 0))
    trajectories = defaultdict(list)
    for o in arch_outputs:
        conds = o.get("conditions", {})
        for field in ["context_retention", "max_tokens", "noise_injection",
                      "info_reorder_intensity", "temperature"]:
            val = conds.get(field)
            if val is not None:
                trajectories[field].append(float(val))
    analysis["operating_condition_trajectories"] = dict(trajectories)

    return analysis


# ---------------------------------------------------------------------------
# Regulatory dynamics (architectured only)
# ---------------------------------------------------------------------------

def regulatory_dynamics(regulator_actions: list[dict]) -> dict:
    """Analyse homeostatic regulator behaviour."""
    actions = extract_regulator_actions(regulator_actions)
    if not actions:
        return {"n_actions": 0}

    actions.sort(key=lambda a: a.get("_seq", 0))

    health_trajectory = [a.get("health", 0.0) for a in actions]
    disruption_trajectory = [a.get("disruption_intensity", 0.0) for a in actions]

    # Per-mechanism adjustment frequency and magnitude.
    mechanism_stats = defaultdict(lambda: {"count": 0, "magnitudes": []})
    for a in actions:
        adjustments = a.get("adjustments", {})
        for mech, val in adjustments.items():
            mechanism_stats[mech]["count"] += 1
            if isinstance(val, (int, float)):
                mechanism_stats[mech]["magnitudes"].append(abs(float(val)))

    mechanism_summary = {}
    for mech, data in mechanism_stats.items():
        mechanism_summary[mech] = {
            "adjustment_count": data["count"],
            "magnitude_stats": compute_summary_stats(data["magnitudes"]),
        }

    # Convergence analysis: look at the final 25% of health values.
    # If the standard deviation of the final quarter is significantly
    # lower than the full trajectory, the system converged.
    convergence = {}
    if len(health_trajectory) >= 8:
        quarter = len(health_trajectory) // 4
        full_std = float(np.std(health_trajectory))
        final_std = float(np.std(health_trajectory[-quarter:]))
        final_mean = float(np.mean(health_trajectory[-quarter:]))

        convergence["full_trajectory_std"] = full_std
        convergence["final_quarter_std"] = final_std
        convergence["final_quarter_mean_health"] = final_mean
        convergence["std_ratio"] = final_std / full_std if full_std > 0 else float("inf")
        convergence["converged"] = final_std < full_std * 0.5
    else:
        convergence["converged"] = None
        convergence["note"] = "insufficient data for convergence analysis (need >= 8 actions)"

    return {
        "n_actions": len(actions),
        "health_trajectory": health_trajectory,
        "disruption_trajectory": disruption_trajectory,
        "health_stats": compute_summary_stats(health_trajectory),
        "disruption_stats": compute_summary_stats(disruption_trajectory),
        "mechanism_summary": mechanism_summary,
        "convergence": convergence,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_boxplots(metrics_by_condition: dict, output_dir: Path):
    """Box plots of each metric dimension by condition."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, metric in enumerate(METRIC_DIMENSIONS):
        ax = axes[i]
        data_to_plot = []
        labels = []
        for cond in CONDITIONS:
            values = metrics_by_condition.get(cond, {}).get(metric, [])
            if values:
                data_to_plot.append(values)
                labels.append(CONDITION_LABELS.get(cond, cond))
            else:
                data_to_plot.append([0])
                labels.append(f"{CONDITION_LABELS.get(cond, cond)}\n(no data)")

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.5})
        colours = ["#4C72B0", "#55A868", "#C44E52"]
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)

        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)

    # Hide the unused subplot.
    if len(METRIC_DIMENSIONS) < len(axes):
        for j in range(len(METRIC_DIMENSIONS), len(axes)):
            axes[j].set_visible(False)

    fig.suptitle("Metric Distributions by Experimental Condition", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "metric_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_regulatory_trajectories(reg_dynamics: dict, output_dir: Path):
    """Health and disruption intensity trajectories for architectured runs."""
    health = reg_dynamics.get("health_trajectory", [])
    disruption = reg_dynamics.get("disruption_trajectory", [])
    if not health:
        print("  Skipping regulatory trajectory plot (no data)")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    steps = range(1, len(health) + 1)
    ax1.plot(steps, health, color="#4C72B0", linewidth=1, alpha=0.7)
    if len(health) >= 5:
        window = max(3, len(health) // 10)
        smoothed = np.convolve(health, np.ones(window) / window, mode="valid")
        ax1.plot(range(window, len(health) + 1), smoothed,
                 color="#C44E52", linewidth=2, label=f"Moving avg (w={window})")
        ax1.legend()
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Health Signal")
    ax1.set_title("Homeostatic Health Signal Over Time")
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(alpha=0.3)

    ax2.plot(steps, disruption, color="#55A868", linewidth=1, alpha=0.7)
    if len(disruption) >= 5:
        window = max(3, len(disruption) // 10)
        smoothed = np.convolve(disruption, np.ones(window) / window, mode="valid")
        ax2.plot(range(window, len(disruption) + 1), smoothed,
                 color="#C44E52", linewidth=2, label=f"Moving avg (w={window})")
        ax2.legend()
    ax2.set_ylabel("Disruption Intensity")
    ax2.set_xlabel("Regulation Step")
    ax2.set_title("Disruption Intensity Over Time")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)

    fig.suptitle("Regulatory Dynamics (Architectured Condition)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "regulatory_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_operating_conditions(trajectories: dict, output_dir: Path):
    """Operating conditions trajectory over the task sequence."""
    if not trajectories:
        print("  Skipping operating conditions plot (no data)")
        return

    fields = [f for f in ["context_retention", "noise_injection",
                          "info_reorder_intensity", "temperature"]
              if f in trajectories and trajectories[f]]
    if not fields:
        print("  Skipping operating conditions plot (no relevant fields)")
        return

    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 3 * len(fields)), sharex=True)
    if len(fields) == 1:
        axes = [axes]

    colours = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for ax, field, colour in zip(axes, fields, colours):
        values = trajectories[field]
        ax.plot(range(1, len(values) + 1), values, color=colour, linewidth=1.5)
        ax.set_ylabel(field.replace("_", " ").title())
        ax.grid(alpha=0.3)
        if field != "max_tokens":
            ax.set_ylim(-0.05, 1.05)

    axes[-1].set_xlabel("Task Sequence Position")
    fig.suptitle("Operating Condition Trajectories (Architectured)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "operating_conditions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_response_length_distributions(output_data: dict, output_dir: Path):
    """Response length distribution by condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colours = {"control": "#4C72B0", "history_only": "#55A868", "architectured": "#C44E52"}
    for cond in CONDITIONS:
        lengths = output_data.get(cond, {}).get("response_lengths", [])
        if not lengths:
            continue
        ax.hist(lengths, bins=30, alpha=0.5, label=CONDITION_LABELS.get(cond, cond),
                color=colours.get(cond, "gray"), edgecolor="white")

    ax.set_xlabel("Response Length (characters)")
    ax.set_ylabel("Frequency")
    ax.set_title("Response Length Distribution by Condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "response_length_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_separator(char="=", width=80):
    print(char * width)


def print_section(title: str):
    print()
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def print_summary_table(summary: dict):
    """Print per-condition summary statistics."""
    print_section("1. PER-CONDITION SUMMARY STATISTICS")
    for metric in METRIC_DIMENSIONS:
        print(f"  {metric.replace('_', ' ').upper()}")
        print(f"  {'Condition':<16} {'N':>5} {'Mean':>7} {'Median':>7} {'SD':>7} {'IQR':>7} {'Min':>7} {'Max':>7}")
        print(f"  {'-'*73}")
        for cond in CONDITIONS:
            s = summary.get(cond, {}).get(metric, {})
            n = s.get("n", 0)
            if n == 0:
                print(f"  {CONDITION_LABELS.get(cond, cond):<16} {'--':>5} {'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>7}")
            else:
                print(f"  {CONDITION_LABELS.get(cond, cond):<16} {n:>5} {s['mean']:>7.3f} {s['median']:>7.3f} "
                      f"{s['std']:>7.3f} {s['iqr']:>7.3f} {s['min']:>7.3f} {s['max']:>7.3f}")
        print()


def print_statistical_tests(test_results: list[dict]):
    """Print pairwise statistical test results."""
    print_section("2. BETWEEN-CONDITION STATISTICAL TESTS (CONFIRMATORY)")
    print("  Test: Mann-Whitney U (two-sided, non-parametric)")
    print("  Effect size: Cliff's delta with bootstrap 95% CI")
    print("  Correction: Bonferroni for multiple comparisons")
    print()

    # Group by scope.
    by_scope = defaultdict(list)
    for r in test_results:
        by_scope[r["scope"]].append(r)

    for scope in sorted(by_scope.keys()):
        scope_results = by_scope[scope]
        label = "POOLED ACROSS SCENARIOS" if scope == "pooled" else f"SCENARIO: {scope}"
        print(f"  {label}")
        print(f"  {'Metric':<22} {'Comparison':<30} {'U':>8} {'p':>10} {'p_corr':>10} "
              f"{'delta':>7} {'95% CI':>16} {'Effect':>11} {'Sig':>5}")
        print(f"  {'-'*130}")

        for r in scope_results:
            comp = f"{CONDITION_LABELS[r['condition_a']]} vs {CONDITION_LABELS[r['condition_b']]}"
            sig = "***" if r["significant_001"] else ("*" if r["significant_005"] else "")
            ci_str = f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]"
            print(f"  {r['metric']:<22} {comp:<30} {r['U_statistic']:>8.0f} "
                  f"{r['p_value']:>10.4f} {r['p_corrected']:>10.4f} "
                  f"{r['cliffs_delta']:>+7.3f} {ci_str:>16} {r['effect_interpretation']:>11} {sig:>5}")
        print()


def print_behavioural_analysis(behaviour: dict):
    """Print behavioural divergence analysis."""
    print_section("3. BEHAVIOURAL DIVERGENCE ANALYSIS (EXPLORATORY)")

    print("  RESPONSE LENGTH (characters)")
    print(f"  {'Condition':<16} {'N':>6} {'Mean':>8} {'Median':>8} {'SD':>8}")
    print(f"  {'-'*50}")
    for cond in CONDITIONS:
        s = behaviour.get("response_lengths", {}).get(cond, {})
        n = s.get("n", 0)
        if n == 0:
            print(f"  {CONDITION_LABELS.get(cond, cond):<16} {'--':>6} {'--':>8} {'--':>8} {'--':>8}")
        else:
            print(f"  {CONDITION_LABELS.get(cond, cond):<16} {n:>6} {s['mean']:>8.0f} {s['median']:>8.0f} {s['std']:>8.0f}")
    print()

    print("  TOKEN USAGE")
    print(f"  {'Condition':<16} {'Input Mean':>11} {'Input Med':>10} {'Output Mean':>12} {'Output Med':>11}")
    print(f"  {'-'*62}")
    for cond in CONDITIONS:
        tu = behaviour.get("token_usage", {}).get(cond, {})
        it_s = tu.get("input_tokens", {})
        ot_s = tu.get("output_tokens", {})
        if it_s.get("n", 0) == 0:
            print(f"  {CONDITION_LABELS.get(cond, cond):<16} {'--':>11} {'--':>10} {'--':>12} {'--':>11}")
        else:
            print(f"  {CONDITION_LABELS.get(cond, cond):<16} {it_s['mean']:>11.0f} {it_s['median']:>10.0f} "
                  f"{ot_s['mean']:>12.0f} {ot_s['median']:>11.0f}")
    print()

    tc = behaviour.get("transition_counts", {})
    if tc:
        print("  STATE TRANSITION COUNTS (architectured only)")
        total = behaviour.get("total_transitions_architectured", 0)
        print(f"  Total transitions: {total}")
        for field, count in sorted(tc.items(), key=lambda x: -x[1]):
            print(f"    {field:<30} {count:>6}")
        print()


def print_regulatory_dynamics(reg: dict):
    """Print regulatory dynamics analysis."""
    print_section("4. REGULATORY DYNAMICS (ARCHITECTURED ONLY)")

    n = reg.get("n_actions", 0)
    print(f"  Total regulatory actions: {n}")
    if n == 0:
        print("  No regulatory data available.")
        return

    hs = reg.get("health_stats", {})
    ds = reg.get("disruption_stats", {})
    print(f"\n  HEALTH SIGNAL")
    print(f"    Mean: {hs.get('mean', 0):.4f}  Median: {hs.get('median', 0):.4f}  "
          f"SD: {hs.get('std', 0):.4f}  Range: [{hs.get('min', 0):.4f}, {hs.get('max', 0):.4f}]")

    print(f"\n  DISRUPTION INTENSITY")
    print(f"    Mean: {ds.get('mean', 0):.4f}  Median: {ds.get('median', 0):.4f}  "
          f"SD: {ds.get('std', 0):.4f}  Range: [{ds.get('min', 0):.4f}, {ds.get('max', 0):.4f}]")

    ms = reg.get("mechanism_summary", {})
    if ms:
        print(f"\n  PER-MECHANISM ADJUSTMENTS")
        print(f"  {'Mechanism':<30} {'Count':>6} {'Mean Mag':>9} {'SD Mag':>9}")
        print(f"  {'-'*56}")
        for mech, data in sorted(ms.items()):
            mag = data.get("magnitude_stats", {})
            if mag.get("n", 0) > 0:
                print(f"  {mech:<30} {data['adjustment_count']:>6} "
                      f"{mag['mean']:>9.4f} {mag['std']:>9.4f}")
            else:
                print(f"  {mech:<30} {data['adjustment_count']:>6} {'--':>9} {'--':>9}")

    conv = reg.get("convergence", {})
    print(f"\n  CONVERGENCE ANALYSIS")
    if conv.get("converged") is None:
        print(f"    {conv.get('note', 'No data')}")
    else:
        print(f"    Full trajectory SD:    {conv.get('full_trajectory_std', 0):.4f}")
        print(f"    Final quarter SD:      {conv.get('final_quarter_std', 0):.4f}")
        print(f"    SD ratio:              {conv.get('std_ratio', 0):.4f}")
        print(f"    Final quarter mean:    {conv.get('final_quarter_mean_health', 0):.4f}")
        converged = conv.get("converged", False)
        print(f"    Converged (SD < 50%):  {'Yes' if converged else 'No'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SUSAN experiment analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python analyse.py --data-dir ../output
  python analyse.py --data-dir ../output --output-dir ./results --no-plots
""")
    parser.add_argument("--data-dir", required=True,
                        help="Path to the experiment output directory containing run subdirectories")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output files (default: same as data-dir)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_ITERATIONS,
                        help=f"Number of bootstrap iterations for CIs (default: {BOOTSTRAP_ITERATIONS})")
    args = parser.parse_args()

    global BOOTSTRAP_ITERATIONS
    BOOTSTRAP_ITERATIONS = args.bootstrap_n

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_dir

    if not data_dir.exists():
        print(f"error: data directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print_section("DATA LOADING")
    all_data = load_all_data(data_dir)

    # ------------------------------------------------------------------
    # Extract and organise
    # ------------------------------------------------------------------
    results = extract_experiment_results(all_data["experiment_results"])
    assessments = extract_monitor_assessments(all_data["monitor_assessments"])
    regulator_actions_raw = all_data["regulator_actions"]
    state_transitions = extract_state_transitions(all_data["state_transitions"])
    core_outputs = extract_core_outputs(all_data["core_outputs"])

    print(f"\n  Extracted {len(results)} experiment results")
    print(f"  Extracted {len(assessments)} monitor assessments")
    print(f"  Extracted {len(core_outputs)} core outputs")
    print(f"  Extracted {len(state_transitions)} state transitions")

    # Collect metrics: from monitor assessments (architectured) and final
    # metrics from experiment results (all conditions).
    assessment_metrics = collect_assessment_metrics(assessments)
    final_metrics = collect_final_metrics(results)

    # For the pooled statistical tests we use final_metrics pooled across
    # scenarios. Build a flat {condition: {metric: [values]}} structure.
    pooled_metrics = {c: defaultdict(list) for c in CONDITIONS}
    for cond in CONDITIONS:
        if cond in final_metrics:
            for scenario in final_metrics[cond]:
                for metric in METRIC_DIMENSIONS:
                    pooled_metrics[cond][metric].extend(
                        final_metrics[cond][scenario].get(metric, []))
        # Also fold in assessment metrics for architectured if we have them
        # and final_metrics are sparse.
        if cond in assessment_metrics:
            for metric in METRIC_DIMENSIONS:
                vals = assessment_metrics[cond].get(metric, [])
                if vals and not pooled_metrics[cond][metric]:
                    pooled_metrics[cond][metric] = vals

    # Discover scenarios.
    all_scenarios = set()
    for cond in CONDITIONS:
        if cond in final_metrics:
            all_scenarios.update(final_metrics[cond].keys())
    scenarios = sorted(all_scenarios)

    # ------------------------------------------------------------------
    # 1. Summary statistics
    # ------------------------------------------------------------------
    summary = build_summary_table(pooled_metrics)
    scenario_summary = build_scenario_summary(final_metrics)
    print_summary_table(summary)

    if scenario_summary:
        print_section("1b. PER-SCENARIO BREAKDOWN")
        for cond in CONDITIONS:
            cond_scenarios = scenario_summary.get(cond, {})
            if not cond_scenarios:
                continue
            print(f"  {CONDITION_LABELS.get(cond, cond).upper()}")
            for scenario in sorted(cond_scenarios.keys()):
                print(f"    Scenario: {scenario}")
                print(f"    {'Metric':<22} {'N':>5} {'Mean':>7} {'Median':>7} {'SD':>7}")
                print(f"    {'-'*50}")
                for metric in METRIC_DIMENSIONS:
                    s = cond_scenarios[scenario].get(metric, {})
                    n = s.get("n", 0)
                    if n == 0:
                        print(f"    {metric:<22} {'--':>5} {'--':>7} {'--':>7} {'--':>7}")
                    else:
                        print(f"    {metric:<22} {n:>5} {s['mean']:>7.3f} {s['median']:>7.3f} {s['std']:>7.3f}")
                print()

    # ------------------------------------------------------------------
    # 2. Statistical tests
    # ------------------------------------------------------------------
    pooled_tests = run_pairwise_tests(pooled_metrics, scenarios)
    scenario_tests = run_scenario_tests(final_metrics, len(pooled_tests))
    all_tests = pooled_tests + scenario_tests
    print_statistical_tests(all_tests)

    # ------------------------------------------------------------------
    # 3. Behavioural divergence
    # ------------------------------------------------------------------
    behaviour = behavioural_analysis(core_outputs, assessments, state_transitions, results)
    print_behavioural_analysis(behaviour)

    # ------------------------------------------------------------------
    # 4. Regulatory dynamics
    # ------------------------------------------------------------------
    reg = regulatory_dynamics(regulator_actions_raw)
    print_regulatory_dynamics(reg)

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print_section("5. GENERATING PLOTS")
        plot_metric_boxplots(pooled_metrics, output_dir)
        plot_regulatory_trajectories(reg, output_dir)
        plot_operating_conditions(
            behaviour.get("operating_condition_trajectories", {}), output_dir)

        output_data = collect_per_task_from_outputs(core_outputs)
        plot_response_length_distributions(
            {c: {"response_lengths": output_data.get(c, {}).get("response_lengths", [])}
             for c in CONDITIONS},
            output_dir)

    # ------------------------------------------------------------------
    # 6. Save JSON summary
    # ------------------------------------------------------------------
    json_output = {
        "summary_statistics": summary,
        "scenario_summary": scenario_summary,
        "statistical_tests": all_tests,
        "behavioural_analysis": {
            k: v for k, v in behaviour.items()
            if k != "operating_condition_trajectories"
        },
        "regulatory_dynamics": {
            k: v for k, v in reg.items()
            if k not in ("health_trajectory", "disruption_trajectory")
        },
        "metadata": {
            "n_experiment_results": len(results),
            "n_monitor_assessments": len(assessments),
            "n_core_outputs": len(core_outputs),
            "n_state_transitions": len(state_transitions),
            "scenarios": scenarios,
            "conditions": CONDITIONS,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_ci": BOOTSTRAP_CI,
            "bonferroni_n_tests": len(all_tests),
        },
    }

    json_path = output_dir / "analysis_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"\n  Saved JSON summary: {json_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_section("ANALYSIS COMPLETE")
    sig_tests = [t for t in all_tests if t.get("significant_005")]
    print(f"  Total statistical tests: {len(all_tests)}")
    print(f"  Significant at p < 0.05 (Bonferroni-corrected): {len(sig_tests)}")
    if sig_tests:
        print()
        for t in sig_tests:
            comp = f"{CONDITION_LABELS[t['condition_a']]} vs {CONDITION_LABELS[t['condition_b']]}"
            print(f"    {t['scope']:<12} {t['metric']:<22} {comp:<30} "
                  f"delta={t['cliffs_delta']:+.3f} p_corr={t['p_corrected']:.4f} ({t['effect_interpretation']})")
    print()


if __name__ == "__main__":
    main()
