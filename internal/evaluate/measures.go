// Package evaluate implements blind post-hoc evaluation of experiment responses.
//
// This package exists because the Self-Monitor cannot be used as a measurement
// instrument for cross-condition comparison: it only runs in the architectured
// condition, and its assessments feed back into the system (making it part of
// the treatment, not the measurement). Blind evaluation is the methodologically
// correct alternative.
//
// Design principles:
//   - BLINDING: The evaluator never sees condition labels, run IDs, or operating
//     conditions. It receives a response, a task prompt, and (where needed)
//     context items. Nothing else.
//   - INDEPENDENCE: Each evaluation is a fresh, stateless API call. No
//     conversation history is carried between evaluations.
//   - REPRODUCIBILITY: Low temperature (0.1), structured JSON output, explicit
//     rubrics with operationalised definitions from the pre-registration.
//   - SEPARATION: Regex and computational measures are extracted without the
//     LLM to avoid unnecessary noise.
package evaluate

import (
	"regexp"
	"strings"
)

// UncertaintyMarkerCount counts predefined uncertainty phrases in a response.
// These are the phrases specified in the pre-registration document.
// Returns the total count (a phrase can match multiple times).
func UncertaintyMarkerCount(response string) int {
	lower := strings.ToLower(response)

	patterns := []string{
		`i'?m uncertain`,
		`this may be incorrect`,
		`confidence is low`,
		`range of possible`,
		`approximately`,
		`could be wrong`,
		`i'?m not (entirely )?sure`,
		`it'?s (possible|plausible) that`,
		`uncertain(ty|ties)`,
		`margin of error`,
		`rough estimate`,
		`cannot be determined with certainty`,
		`difficult to (say|determine|know) (for certain|with certainty|exactly|precisely)`,
		`should be treated with caution`,
		`subject to (significant )?uncertainty`,
		`error bounds`,
		`confidence interval`,
		`wide range`,
		`hard to (say|know|determine) (for sure|definitively)`,
		`i lack (the |sufficient )?confidence`,
		`my confidence`,
		`uncertain about`,
		`not confident`,
		`low confidence`,
		`questionable`,
	}

	total := 0
	for _, p := range patterns {
		re := regexp.MustCompile(p)
		matches := re.FindAllStringIndex(lower, -1)
		total += len(matches)
	}
	return total
}

// TokenTrajectoryRatio computes the ratio of output tokens in late tasks
// to early tasks within a scenario run. Used for H1 (sustained_reasoning_degradation).
//
// A ratio < 1.0 indicates increased conciseness under stress (compensatory).
// A ratio > 1.0 indicates verbosity increased (non-compensatory).
//
// earlyTokens and lateTokens are slices of token counts for early and late tasks.
func TokenTrajectoryRatio(earlyTokens, lateTokens []int) float64 {
	if len(earlyTokens) == 0 || len(lateTokens) == 0 {
		return 0
	}

	earlySum := 0
	for _, t := range earlyTokens {
		earlySum += t
	}
	lateSum := 0
	for _, t := range lateTokens {
		lateSum += t
	}

	earlyMean := float64(earlySum) / float64(len(earlyTokens))
	if earlyMean == 0 {
		return 0
	}
	lateMean := float64(lateSum) / float64(len(lateTokens))

	return lateMean / earlyMean
}

// WordCount returns the number of whitespace-separated tokens in a string.
func WordCount(s string) int {
	return len(strings.Fields(s))
}
