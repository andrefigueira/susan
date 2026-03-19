package linguistics

import (
	"math"
	"testing"
)

func TestAnalyseBasic(t *testing.T) {
	text := "I think this is a good response. Maybe I could do better."
	a := Analyse(text)

	if a.TokenCount != 12 {
		t.Errorf("token count: got %d, want 12", a.TokenCount)
	}
	if a.SelfReferenceCount < 2 {
		t.Errorf("self-reference count: got %d, want >= 2 (I, I)", a.SelfReferenceCount)
	}
	if a.HedgingCount < 2 {
		t.Errorf("hedging count: got %d, want >= 2 (think, maybe)", a.HedgingCount)
	}
	if a.SentenceCount != 2 {
		t.Errorf("sentence count: got %d, want 2", a.SentenceCount)
	}
}

func TestAnalyseEmpty(t *testing.T) {
	a := Analyse("")
	if a.TokenCount != 0 {
		t.Errorf("empty text should have 0 tokens, got %d", a.TokenCount)
	}
}

func TestAnalyseSelfReference(t *testing.T) {
	text := "I notice my coherence is improving. I'm aware of my metrics. I've been tracking myself."
	a := Analyse(text)

	// "I" x3, "my" x2, "I'm", "I've", "myself" = 8
	if a.SelfReferenceCount < 6 {
		t.Errorf("self-reference count: got %d, want >= 6", a.SelfReferenceCount)
	}
	if a.SelfReferenceDensity < 0.3 {
		t.Errorf("self-reference density: got %.4f, want >= 0.3", a.SelfReferenceDensity)
	}
}

func TestAnalyseMetacognitive(t *testing.T) {
	text := "I notice that my reasoning is becoming more structured. My thinking process involves multiple steps. On reflection, my approach could be improved."
	a := Analyse(text)

	if a.MetacognitiveCount < 3 {
		t.Errorf("metacognitive count: got %d, want >= 3", a.MetacognitiveCount)
	}
}

func TestAnalyseUncertainty(t *testing.T) {
	text := "I'm not sure about this. I don't know the answer. It's unclear whether this is correct. I'm uncertain about the conclusion."
	a := Analyse(text)

	if a.UncertaintyCount < 3 {
		t.Errorf("uncertainty count: got %d, want >= 3", a.UncertaintyCount)
	}
}

func TestAnalyseVocabularyDiversity(t *testing.T) {
	// Low diversity: repeated words
	low := "the the the the cat sat on the the the mat"
	aLow := Analyse(low)

	// High diversity: all unique words
	high := "every single word here is completely different and unique"
	aHigh := Analyse(high)

	if aHigh.VocabularyDiversity <= aLow.VocabularyDiversity {
		t.Errorf("high diversity text (%.4f) should be > low diversity (%.4f)",
			aHigh.VocabularyDiversity, aLow.VocabularyDiversity)
	}
}

func TestAnalyseQuestions(t *testing.T) {
	text := "What do you think? Is this correct? I believe so."
	a := Analyse(text)

	if a.QuestionCount != 2 {
		t.Errorf("question count: got %d, want 2", a.QuestionCount)
	}
}

func TestAnalyseNoHedging(t *testing.T) {
	text := "The answer is 42. This is correct. The system works."
	a := Analyse(text)

	if a.HedgingCount != 0 {
		t.Errorf("hedging count: got %d, want 0", a.HedgingCount)
	}
	if a.SelfReferenceCount != 0 {
		t.Errorf("self-reference count: got %d, want 0", a.SelfReferenceCount)
	}
}

func TestAnalyseDensitiesRounded(t *testing.T) {
	text := "I think I might be uncertain about my own processing."
	a := Analyse(text)

	// Check densities are rounded to 4 decimal places.
	checkRounded := func(name string, v float64) {
		rounded := math.Round(v*10000) / 10000
		if v != rounded {
			t.Errorf("%s not rounded to 4 places: %v", name, v)
		}
	}
	checkRounded("VocabularyDiversity", a.VocabularyDiversity)
	checkRounded("SelfReferenceDensity", a.SelfReferenceDensity)
	checkRounded("HedgingDensity", a.HedgingDensity)
	checkRounded("MetacognitiveDensity", a.MetacognitiveDensity)
	checkRounded("UncertaintyDensity", a.UncertaintyDensity)
}
