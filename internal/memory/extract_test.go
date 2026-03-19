package memory

import "testing"

func TestExtractSelfObservations(t *testing.T) {
	tests := []struct {
		name     string
		response string
		maxObs   int
		wantMin  int
		wantMax  int
	}{
		{
			name:     "no self-reference",
			response: "The supply chain analysis shows three key bottlenecks. First, raw material sourcing is delayed by 2 weeks on average.",
			maxObs:   3,
			wantMin:  0,
			wantMax:  0,
		},
		{
			name:     "single self-reference",
			response: "I notice my coherence has been declining over the last few tasks. The analysis suggests that factor A is dominant.",
			maxObs:   3,
			wantMin:  1,
			wantMax:  1,
		},
		{
			name:     "multiple self-references",
			response: "I notice my coherence is at 0.72. My reasoning depth seems to increase when discussing technical topics. My metrics show a stable pattern overall. The answer to your question is 42.",
			maxObs:   3,
			wantMin:  3,
			wantMax:  3,
		},
		{
			name:     "respects max",
			response: "I notice my coherence is low. My metrics show decline. My reasoning depth drops. My alignment is good. I'm aware of the pattern.",
			maxObs:   2,
			wantMin:  2,
			wantMax:  2,
		},
		{
			name:     "empty response",
			response: "",
			maxObs:   3,
			wantMin:  0,
			wantMax:  0,
		},
		{
			name:     "case insensitive",
			response: "MY COHERENCE seems to be improving over time. The data supports this conclusion.",
			maxObs:   3,
			wantMin:  1,
			wantMax:  1,
		},
		{
			name:     "experiencing marker",
			response: "I'm experiencing what appears to be increased noise in my inputs. This makes the analysis more challenging.",
			maxObs:   3,
			wantMin:  1,
			wantMax:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obs := ExtractSelfObservations(tt.response, tt.maxObs)
			if len(obs) < tt.wantMin || len(obs) > tt.wantMax {
				t.Errorf("got %d observations, want [%d, %d]: %v", len(obs), tt.wantMin, tt.wantMax, obs)
			}
		})
	}
}

func TestExtractSelfObservations_Truncation(t *testing.T) {
	// Build a response with a very long self-referential sentence.
	long := "I notice my coherence is " + string(make([]byte, 400))
	obs := ExtractSelfObservations(long, 1)
	if len(obs) == 0 {
		t.Fatal("expected at least one observation")
	}
	if len(obs[0]) > 303 { // 300 + "..."
		t.Errorf("observation not truncated: length %d", len(obs[0]))
	}
}

func TestSplitSentences(t *testing.T) {
	text := "First sentence. Second sentence! Third sentence? Fourth."
	sentences := splitSentences(text)
	if len(sentences) < 3 {
		t.Errorf("expected at least 3 sentences, got %d: %v", len(sentences), sentences)
	}
}
