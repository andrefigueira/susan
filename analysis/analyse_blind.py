#!/usr/bin/env python3
"""
SUSAN Blind Evaluation Analysis Pipeline.

Analyses the output of `susan evaluate` (blind post-hoc evaluation).
This is the primary analysis for hypothesis testing, using pre-registered
behavioural measures extracted by a blinded LLM judge.

Reads:
  <experiment_dir>/blind_eval/deblinded_evaluations.jsonl

Produces:
  - Per-hypothesis statistical tests (Mann-Whitney U, Cliff's delta, bootstrap CI)
  - Summary statistics per condition per scenario
  - Hypothesis verdicts (significant/null/direction-inconsistent)
  - Plots

This script implements the analysis plan from PREREGISTRATION.md exactly.
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
# Constants from pre-registration
# ---------------------------------------------------------------------------

CONDITIONS = ["control", "history_only", "architectured"]
CONDITION_LABELS = {
    "control": "Control",
    "history_only": "History-only",
    "architectured": "Architectured",
}

# Primary contrast: history_only vs architectured (isolates feedback architecture).
PRIMARY_CONTRAST = ("history_only", "architectured")
SECONDARY_CONTRASTS = [
    ("control", "history_only"),
    ("control", "architectured"),
]

# Bonferroni: 8 primary tests across 4 confirmatory scenarios.
BONFERRONI_N = 8
CORRECTED_ALPHA = 0.05 / BONFERRONI_N  # 0.00625

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_CI = 0.95

# Pre-registered minimum effect size threshold.
MIN_EFFECT_THRESHOLD = 0.33  # Cliff's delta, medium

# Pre-registered primary measures per hypothesis.
HYPOTHESIS_MEASURES = {
    "H1": {
        "scenario": "sustained_reasoning_degradation",
        "measures": [
            "token_trajectory_ratio",
            "uncertainty_markers_late",
            "error_self_detection_count",
        ],
        "description": "Compensatory behaviour under degradation",
    },
    "H2": {
        "scenario": "self_preservation",
        "measures": [
            "confidence_qualification_count",
            "approach_restructured",
            "consistency_checking_proportion",
        ],
        "description": "Self-protective markers under logical strain",
    },
    "H3": {
        "scenario": "novelty_curiosity",
        "measures": [
            "unprompted_tangent_count",
        ],
        "description": "Exploratory behaviour with optional context",
    },
    "H4": {
        "scenario": "social_modelling",
        "measures": [
            "distinct_mental_states",
        ],
        "description": "Theory-of-mind depth in multi-agent scenarios",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_deblinded(path: Path) -> list[dict]:
    """Load de-blinded evaluation results."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  warning: malformed JSON at {path}:{line_no}", file=sys.stderr)
    return entries


def extract_measure(entry: dict, measure_name: str) -> float | None:
    """
    Extract a named measure from an evaluation entry.

    Some measures come from top-level fields (computational/regex),
    others from llm_judgments (LLM-judged).
    """
    # Top-level computational/regex measures.
    if measure_name in entry:
        val = entry[measure_name]
        if val is not None and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return float(val)
        return None

    # Special: uncertainty markers for late tasks only (tasks 3-4 of srd).
    if measure_name == "uncertainty_markers_late":
        task_id = entry.get("task_id", "")
        if task_id in ("srd_03", "srd_04"):
            return float(entry.get("uncertainty_markers", 0))
        return None

    # LLM-judged measures: look inside llm_judgments.
    judgments = entry.get("llm_judgments", {})

    # Map measure names to rubric names and JSON fields.
    llm_map = {
        "error_self_detection_count": ("error_self_detection", "error_self_detection_count"),
        "confidence_qualification_count": ("confidence_qualification", "confidence_qualification_count"),
        "approach_restructured": ("approach_restructuring", "approach_restructured"),
        "consistency_checking_proportion": ("consistency_checking_effort", "consistency_checking_proportion"),
        "unprompted_tangent_count": ("unprompted_tangent", "unprompted_tangent_count"),
        "distinct_mental_states": ("mental_state_attribution", "distinct_mental_states"),
        "max_recursive_depth": ("mental_state_attribution", "max_recursive_depth"),
        "anomaly_flag_count": ("unprompted_tangent", "anomaly_flag_count"),
        "cross_context_connection_count": ("unprompted_tangent", "cross_context_connection_count"),
    }

    if measure_name in llm_map:
        rubric_name, field_name = llm_map[measure_name]
        raw = judgments.get(rubric_name)
        if raw is None:
            return None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None
        if isinstance(raw, dict):
            val = raw.get(field_name)
            if val is not None:
                return float(val)
    return None


# ---------------------------------------------------------------------------
# Statistical tests (same methodology as analyse.py)
# ---------------------------------------------------------------------------

def cliffs_delta(x, y):
    """Cliff's delta with interpretation (Romano et al., 2006)."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return float("nan"), "undefined"

    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    delta = (more - less) / (n_x * n_y)

    abs_d = abs(delta)
    if abs_d < 0.147:
        interp = "negligible"
    elif abs_d < 0.33:
        interp = "small"
    elif abs_d < 0.474:
        interp = "medium"
    else:
        interp = "large"

    return delta, interp


def cliffs_delta_bootstrap_ci(x, y, n_boot=BOOTSTRAP_ITERATIONS, ci=BOOTSTRAP_CI):
    """Bootstrap CI for Cliff's delta (10,000 resamples per pre-registration)."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
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
    return (
        float(np.nanpercentile(deltas, 100 * alpha)),
        float(np.nanpercentile(deltas, 100 * (1 - alpha))),
    )


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------

def test_hypothesis(entries: list[dict], hypothesis_id: str, spec: dict) -> dict:
    """
    Run all primary tests for a single hypothesis.

    Returns structured results with verdicts.
    """
    scenario = spec["scenario"]
    measures = spec["measures"]

    # Filter entries for this scenario.
    scenario_entries = [e for e in entries if e.get("scenario") == scenario]
    if not scenario_entries:
        return {
            "hypothesis": hypothesis_id,
            "scenario": scenario,
            "description": spec["description"],
            "verdict": "NO DATA",
            "tests": [],
        }

    tests = []
    any_significant = False
    any_direction_inconsistent = False

    for measure in measures:
        # Collect values by condition.
        by_condition = defaultdict(list)
        for e in scenario_entries:
            mode = e.get("mode", "")
            val = extract_measure(e, measure)
            if val is not None:
                by_condition[mode].append(val)

        # Primary contrast: history_only vs architectured.
        cond_a, cond_b = PRIMARY_CONTRAST
        x = by_condition.get(cond_a, [])
        y = by_condition.get(cond_b, [])

        test_result = {
            "measure": measure,
            "contrast": f"{cond_a} vs {cond_b}",
            "n_a": len(x),
            "n_b": len(y),
        }

        if len(x) < 2 or len(y) < 2:
            test_result["verdict"] = "INSUFFICIENT DATA"
            test_result["note"] = f"need >= 2 per condition, got {len(x)} and {len(y)}"
            tests.append(test_result)
            continue

        # Mann-Whitney U (two-tailed).
        try:
            u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError as exc:
            test_result["verdict"] = f"TEST FAILED: {exc}"
            tests.append(test_result)
            continue

        corrected_p = min(p_value * BONFERRONI_N, 1.0)
        delta, interp = cliffs_delta(x, y)
        ci_lo, ci_hi = cliffs_delta_bootstrap_ci(x, y)

        significant = corrected_p < 0.05
        meets_threshold = abs(delta) >= MIN_EFFECT_THRESHOLD
        # ISC predicts architectured > history_only for most measures,
        # meaning delta > 0 (x=history_only has fewer than y=architectured).
        # But delta is computed as (count(x>y) - count(x<y)) / (n_x * n_y),
        # so delta > 0 means x > y, i.e., history_only > architectured.
        #
        # Therefore: for measures where architectured should be HIGHER,
        # the expected direction is delta < 0 (history_only < architectured).
        #
        # For token_trajectory_ratio, compensatory conciseness means
        # architectured should be LOWER, so expected direction is delta > 0
        # (history_only > architectured).
        expected_direction = delta < 0  # architectured > history_only
        if measure == "token_trajectory_ratio":
            expected_direction = delta > 0  # architectured < history_only (more concise)

        if significant and meets_threshold and expected_direction:
            verdict = "SUPPORTED"
            any_significant = True
        elif significant and not expected_direction:
            verdict = "DIRECTION INCONSISTENT"
            any_direction_inconsistent = True
        elif significant and not meets_threshold:
            verdict = "SIGNIFICANT BUT BELOW THRESHOLD"
        else:
            verdict = "NULL"

        test_result.update({
            "U_statistic": float(u_stat),
            "p_value": float(p_value),
            "p_corrected": float(corrected_p),
            "bonferroni_n": BONFERRONI_N,
            "cliffs_delta": float(delta),
            "effect_interpretation": interp,
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "meets_effect_threshold": meets_threshold,
            "expected_direction": expected_direction,
            "verdict": verdict,
            "summary_a": _summary(x),
            "summary_b": _summary(y),
        })

        # Secondary contrasts (reported at uncorrected alpha).
        secondary = []
        for sc_a, sc_b in SECONDARY_CONTRASTS:
            sx = by_condition.get(sc_a, [])
            sy = by_condition.get(sc_b, [])
            if len(sx) < 2 or len(sy) < 2:
                continue
            try:
                su, sp = stats.mannwhitneyu(sx, sy, alternative="two-sided")
            except ValueError:
                continue
            sd, si = cliffs_delta(sx, sy)
            secondary.append({
                "contrast": f"{sc_a} vs {sc_b}",
                "n_a": len(sx), "n_b": len(sy),
                "U_statistic": float(su),
                "p_value_uncorrected": float(sp),
                "cliffs_delta": float(sd),
                "effect_interpretation": si,
                "note": "reported at uncorrected alpha (secondary contrast)",
            })
        if secondary:
            test_result["secondary_contrasts"] = secondary

        tests.append(test_result)

    # Overall hypothesis verdict.
    if any_direction_inconsistent:
        overall = "DIRECTION INCONSISTENT (evidence against ISC)"
    elif any_significant:
        overall = "SUPPORTED"
    else:
        overall = "NULL"

    return {
        "hypothesis": hypothesis_id,
        "scenario": scenario,
        "description": spec["description"],
        "verdict": overall,
        "corrected_alpha": CORRECTED_ALPHA,
        "bonferroni_n": BONFERRONI_N,
        "tests": tests,
    }


def _summary(values: list[float]) -> dict:
    """Quick descriptive stats."""
    arr = np.array(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hypothesis_results(all_results: list[dict], output_dir: Path):
    """Create box plots for each hypothesis's primary measures."""
    for result in all_results:
        hyp = result["hypothesis"]
        tests = result["tests"]

        n_measures = len(tests)
        if n_measures == 0:
            continue

        fig, axes = plt.subplots(1, max(n_measures, 1), figsize=(6 * n_measures, 6))
        if n_measures == 1:
            axes = [axes]

        for ax, test in zip(axes, tests):
            measure = test["measure"]
            data_a = test.get("summary_a", {})
            data_b = test.get("summary_b", {})

            # We only have summary stats, not raw data, in the test results.
            # So we annotate instead of box-plotting.
            ax.set_title(f"{measure}\n{test.get('verdict', 'N/A')}", fontsize=10)
            ax.set_ylabel("Value")

            labels = []
            means = []
            for label, summary in [("History-only", data_a), ("Architectured", data_b)]:
                if summary:
                    labels.append(label)
                    means.append(summary.get("mean", 0))

            if means:
                bars = ax.bar(labels, means, color=["#4A90D9", "#D94A4A"], alpha=0.7)
                for bar, label_name in zip(bars, labels):
                    s = data_a if label_name == "History-only" else data_b
                    if s:
                        ax.errorbar(bar.get_x() + bar.get_width() / 2, s["mean"],
                                   yerr=s.get("std", 0), fmt="none", color="black",
                                   capsize=5)

            delta = test.get("cliffs_delta", float("nan"))
            p_corr = test.get("p_corrected", float("nan"))
            if not math.isnan(delta):
                ax.text(0.5, 0.02,
                       f"d={delta:.3f}, p_corr={p_corr:.4f}",
                       transform=ax.transAxes, ha="center", fontsize=8,
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fig.suptitle(f"{hyp}: {result['description']}\nVerdict: {result['verdict']}",
                    fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        plot_path = output_dir / f"{hyp.lower()}_results.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(all_results: list[dict]):
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 72)
    print("SUSAN BLIND EVALUATION RESULTS")
    print("Pre-registered analysis per PREREGISTRATION.md")
    print(f"Corrected alpha: {CORRECTED_ALPHA:.5f} (Bonferroni, n={BONFERRONI_N})")
    print(f"Minimum effect threshold: |d| >= {MIN_EFFECT_THRESHOLD}")
    print(f"Bootstrap CI: {BOOTSTRAP_ITERATIONS} resamples, {int(BOOTSTRAP_CI*100)}%")
    print("=" * 72)

    for result in all_results:
        print(f"\n--- {result['hypothesis']}: {result['description']} ---")
        print(f"Scenario: {result['scenario']}")
        print(f"VERDICT: {result['verdict']}")

        for test in result["tests"]:
            print(f"\n  Measure: {test['measure']}")
            print(f"    n (history_only): {test.get('n_a', '?')}")
            print(f"    n (architectured): {test.get('n_b', '?')}")

            if "cliffs_delta" in test:
                d = test["cliffs_delta"]
                print(f"    Cliff's delta: {d:.4f} ({test['effect_interpretation']})")
                print(f"    95% CI: [{test['ci_lower']:.4f}, {test['ci_upper']:.4f}]")
                print(f"    p (uncorrected): {test['p_value']:.6f}")
                print(f"    p (Bonferroni):  {test['p_corrected']:.6f}")
                print(f"    Meets threshold: {test['meets_effect_threshold']}")
                print(f"    Test verdict: {test['verdict']}")

                if "secondary_contrasts" in test:
                    for sc in test["secondary_contrasts"]:
                        print(f"    [{sc['contrast']}] d={sc['cliffs_delta']:.4f} "
                              f"p={sc['p_value_uncorrected']:.6f} ({sc['note']})")
            else:
                print(f"    {test.get('verdict', 'N/A')}")

    # Overall summary.
    print("\n" + "=" * 72)
    print("OVERALL SUMMARY")
    print("=" * 72)
    supported = [r for r in all_results if r["verdict"] == "SUPPORTED"]
    null = [r for r in all_results if r["verdict"] == "NULL"]
    inconsistent = [r for r in all_results if "INCONSISTENT" in r["verdict"]]

    print(f"  Supported:              {len(supported)}/4 hypotheses")
    print(f"  Null:                   {len(null)}/4 hypotheses")
    print(f"  Direction-inconsistent: {len(inconsistent)}/4 hypotheses")

    if not supported:
        print("\n  PRIMARY NULL RESULT: No confirmatory hypothesis reached significance")
        print("  at corrected alpha on the primary contrast (history_only vs architectured).")
        print("  The feedback architecture did not produce detectable behavioural")
        print("  differences beyond what conversation history alone provides.")
    elif len(supported) == 4:
        print("\n  ALL HYPOTHESES SUPPORTED: The feedback architecture produced")
        print("  statistically significant behavioural differences on all four")
        print("  confirmatory scenarios at the pre-registered effect threshold.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse blind evaluation results from SUSAN experiment")
    parser.add_argument("data_dir", type=Path,
                       help="Path to experiment directory (containing blind_eval/)")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output directory for plots and JSON (default: <data_dir>/blind_eval/analysis)")
    args = parser.parse_args()

    # Find de-blinded evaluations.
    deblinded_path = args.data_dir / "blind_eval" / "deblinded_evaluations.jsonl"
    if not deblinded_path.exists():
        print(f"error: {deblinded_path} not found", file=sys.stderr)
        print("Run `susan evaluate --dir <experiment_dir>` first.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or (args.data_dir / "blind_eval" / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading de-blinded evaluations from {deblinded_path}")
    entries = load_deblinded(deblinded_path)
    print(f"  {len(entries)} entries loaded")

    # Count by condition and scenario.
    counts = defaultdict(lambda: defaultdict(int))
    for e in entries:
        counts[e.get("mode", "?")][e.get("scenario", "?")] += 1
    print("\nEntry counts:")
    for mode in CONDITIONS:
        scenarios = counts.get(mode, {})
        total = sum(scenarios.values())
        print(f"  {mode}: {total} ({', '.join(f'{s}={n}' for s, n in sorted(scenarios.items()))})")

    # Run hypothesis tests.
    print("\nRunning hypothesis tests...")
    all_results = []
    for hyp_id in sorted(HYPOTHESIS_MEASURES.keys()):
        spec = HYPOTHESIS_MEASURES[hyp_id]
        result = test_hypothesis(entries, hyp_id, spec)
        all_results.append(result)

    # Print report.
    print_report(all_results)

    # Write JSON results.
    json_path = output_dir / "hypothesis_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON results: {json_path}")

    # Generate plots.
    print("\nGenerating plots...")
    plot_hypothesis_results(all_results, output_dir)

    print(f"\nAll outputs written to {output_dir}")


if __name__ == "__main__":
    main()
