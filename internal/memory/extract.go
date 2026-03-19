package memory

import (
	"strings"
	"unicode"
)

// selfReferentialMarkers are phrases that indicate SUSAN is commenting on
// her own internal state. These are matched case-insensitively against
// individual sentences in her responses.
var selfReferentialMarkers = []string{
	"i notice my",
	"i notice that my",
	"i observe my",
	"i observe that my",
	"my coherence",
	"my metrics",
	"my reasoning depth",
	"my alignment",
	"my operating conditions",
	"my disruption",
	"i'm experiencing",
	"i am experiencing",
	"my internal state",
	"my feedback loop",
	"system status",
	"my performance",
	"i can see that my",
	"i can feel",
	"my trend",
	"i'm aware",
	"i am aware",
}

// ExtractSelfObservations scans a response for sentences where SUSAN
// comments on her own metrics or internal state. Returns up to maxObs
// matching sentences.
func ExtractSelfObservations(response string, maxObs int) []string {
	if maxObs <= 0 {
		maxObs = 3
	}

	sentences := splitSentences(response)
	var observations []string

	for _, s := range sentences {
		s = strings.TrimSpace(s)
		if len(s) < 10 {
			continue
		}
		lower := strings.ToLower(s)
		for _, marker := range selfReferentialMarkers {
			if strings.Contains(lower, marker) {
				// Truncate long observations.
				if len(s) > 300 {
					s = s[:297] + "..."
				}
				observations = append(observations, s)
				break
			}
		}
		if len(observations) >= maxObs {
			break
		}
	}

	return observations
}

// splitSentences splits text into sentences at period, exclamation, or
// question mark boundaries. Good enough for extraction, not a real NLP
// sentence splitter.
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	for i, r := range text {
		current.WriteRune(r)
		if (r == '.' || r == '!' || r == '?') && i+1 < len(text) {
			next := rune(text[i+1])
			if unicode.IsSpace(next) || unicode.IsUpper(next) {
				sentences = append(sentences, current.String())
				current.Reset()
			}
		}
	}
	if current.Len() > 0 {
		sentences = append(sentences, current.String())
	}
	return sentences
}
