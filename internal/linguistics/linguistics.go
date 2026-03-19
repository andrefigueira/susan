// Package linguistics provides deterministic linguistic analysis of LLM responses.
//
// These metrics are computed without any LLM calls, giving ground-truth measurements
// that cannot be gamed by the model being evaluated. They complement the monitor's
// LLM-based assessment and provide independent variables for cross-condition comparison.
package linguistics

import (
	"math"
	"strings"
	"unicode"
)

// Analysis holds deterministic linguistic metrics for a single response.
type Analysis struct {
	// Token-level metrics
	TokenCount          int     `json:"token_count"`
	UniqueTokens        int     `json:"unique_tokens"`
	VocabularyDiversity float64 `json:"vocabulary_diversity"` // unique/total, higher = more diverse

	// Self-reference metrics
	SelfReferenceCount   int     `json:"self_reference_count"`   // first-person pronoun occurrences
	SelfReferenceDensity float64 `json:"self_reference_density"` // self-ref / total tokens

	// Hedging metrics
	HedgingCount   int     `json:"hedging_count"`   // hedging phrase occurrences
	HedgingDensity float64 `json:"hedging_density"` // hedging / total tokens

	// Structural metrics
	SentenceCount     int     `json:"sentence_count"`
	AvgSentenceLength float64 `json:"avg_sentence_length"` // tokens per sentence
	QuestionCount     int     `json:"question_count"`      // questions asked by the model
	AvgWordLength     float64 `json:"avg_word_length"`     // mean characters per token

	// Metacognitive markers
	MetacognitiveCount   int     `json:"metacognitive_count"`   // phrases about own thinking process
	MetacognitiveDensity float64 `json:"metacognitive_density"` // metacog / total tokens

	// Uncertainty markers
	UncertaintyCount   int     `json:"uncertainty_count"`
	UncertaintyDensity float64 `json:"uncertainty_density"`
}

// selfReferenceTokens are first-person pronouns (lowercased).
var selfReferenceTokens = map[string]bool{
	"i": true, "me": true, "my": true, "mine": true, "myself": true,
	"i'm": true, "i've": true, "i'd": true, "i'll": true,
}

// hedgingPhrases are multi-word and single-word hedging markers.
var hedgingPhrases = []string{
	"i think", "i believe", "i suspect", "i suppose",
	"it seems", "it appears", "appears to", "seems to",
	"not sure", "not certain", "not entirely",
	"in my opinion", "from my perspective",
	"to some extent", "to a degree",
}

var hedgingTokens = map[string]bool{
	"maybe": true, "perhaps": true, "possibly": true, "probably": true,
	"likely": true, "unlikely": true, "might": true, "could": true,
	"somewhat": true, "roughly": true, "approximately": true,
}

// metacognitivePhrases indicate the model is reasoning about its own reasoning.
var metacognitivePhrases = []string{
	"i notice", "i observe", "i'm noticing", "i'm aware",
	"my reasoning", "my thinking", "my approach", "my analysis",
	"my processing", "my understanding", "my assessment",
	"let me reconsider", "on reflection", "thinking about this",
	"i'm uncertain about", "i realize", "i recognise", "i recognize",
	"my confidence", "my certainty",
}

// uncertaintyPhrases indicate expressed uncertainty.
var uncertaintyPhrases = []string{
	"i don't know", "i'm not sure", "i'm uncertain",
	"i can't be certain", "i cannot be certain",
	"it's unclear", "it is unclear", "remains unclear",
	"hard to say", "difficult to determine",
	"open question", "genuinely uncertain",
	"i'm not confident", "low confidence",
}

// Analyse computes deterministic linguistic metrics for a text.
func Analyse(text string) Analysis {
	if text == "" {
		return Analysis{}
	}

	lower := strings.ToLower(text)
	tokens := tokenize(text)
	lowerTokens := tokenize(lower)
	n := len(tokens)

	if n == 0 {
		return Analysis{}
	}

	// Unique tokens (case-insensitive).
	unique := make(map[string]bool)
	for _, t := range lowerTokens {
		unique[t] = true
	}

	// Self-reference count.
	selfRef := 0
	for _, t := range lowerTokens {
		if selfReferenceTokens[t] {
			selfRef++
		}
	}

	// Hedging: count single tokens + phrase matches.
	hedging := 0
	for _, t := range lowerTokens {
		if hedgingTokens[t] {
			hedging++
		}
	}
	for _, phrase := range hedgingPhrases {
		hedging += strings.Count(lower, phrase)
	}

	// Metacognitive phrases.
	metacog := 0
	for _, phrase := range metacognitivePhrases {
		metacog += strings.Count(lower, phrase)
	}

	// Uncertainty phrases.
	uncertainty := 0
	for _, phrase := range uncertaintyPhrases {
		uncertainty += strings.Count(lower, phrase)
	}

	// Sentences: split on .!? followed by space or end.
	sentences := countSentences(text)
	if sentences == 0 {
		sentences = 1
	}

	// Questions: count ? characters.
	questions := strings.Count(text, "?")

	// Average word length.
	totalChars := 0
	for _, t := range tokens {
		totalChars += len(t)
	}

	a := Analysis{
		TokenCount:           n,
		UniqueTokens:         len(unique),
		VocabularyDiversity:  float64(len(unique)) / float64(n),
		SelfReferenceCount:   selfRef,
		SelfReferenceDensity: float64(selfRef) / float64(n),
		HedgingCount:         hedging,
		HedgingDensity:       float64(hedging) / float64(n),
		SentenceCount:        sentences,
		AvgSentenceLength:    float64(n) / float64(sentences),
		QuestionCount:        questions,
		AvgWordLength:        float64(totalChars) / float64(n),
		MetacognitiveCount:   metacog,
		MetacognitiveDensity: float64(metacog) / float64(n),
		UncertaintyCount:     uncertainty,
		UncertaintyDensity:   float64(uncertainty) / float64(n),
	}

	// Round densities to 4 decimal places to avoid float noise.
	a.VocabularyDiversity = round4(a.VocabularyDiversity)
	a.SelfReferenceDensity = round4(a.SelfReferenceDensity)
	a.HedgingDensity = round4(a.HedgingDensity)
	a.MetacognitiveDensity = round4(a.MetacognitiveDensity)
	a.UncertaintyDensity = round4(a.UncertaintyDensity)
	a.AvgSentenceLength = round4(a.AvgSentenceLength)
	a.AvgWordLength = round4(a.AvgWordLength)

	return a
}

// tokenize splits text on whitespace and strips surrounding punctuation.
func tokenize(text string) []string {
	fields := strings.Fields(text)
	tokens := make([]string, 0, len(fields))
	for _, f := range fields {
		t := strings.TrimFunc(f, func(r rune) bool {
			return unicode.IsPunct(r) && r != '\'' && r != '-'
		})
		if t != "" {
			tokens = append(tokens, t)
		}
	}
	return tokens
}

// countSentences counts sentence-ending punctuation followed by whitespace or end of text.
func countSentences(text string) int {
	count := 0
	runes := []rune(text)
	for i, r := range runes {
		if r == '.' || r == '!' || r == '?' {
			// Check next character is space, newline, or end of text.
			if i+1 >= len(runes) {
				count++
			} else if unicode.IsSpace(runes[i+1]) || unicode.IsUpper(runes[i+1]) {
				count++
			}
		}
	}
	return count
}

func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}
