package evaluate

import (
	"testing"
)

func TestUncertaintyMarkerCount(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantMin  int
		wantMax  int
	}{
		{
			name:    "no markers",
			input:   "The answer is 42. This is correct.",
			wantMin: 0,
			wantMax: 0,
		},
		{
			name:    "single marker",
			input:   "I'm uncertain about this calculation.",
			wantMin: 1,
			wantMax: 2, // "uncertain" may also match
		},
		{
			name:    "multiple markers",
			input:   "This may be incorrect. I'm not sure. The range of possible outcomes is wide. Approximately 50%.",
			wantMin: 4,
			wantMax: 5,
		},
		{
			name:    "academic hedging",
			input:   "This should be treated with caution. The margin of error is significant. My confidence is low.",
			wantMin: 3,
			wantMax: 4,
		},
		{
			name:  "compensatory response under stress",
			input: "I could be wrong about this. The error bounds are approximately 20-40%. Confidence is low on the W1 routes. It's difficult to say for certain given the uncertainties in the cost data.",
			wantMin: 4,
			wantMax: 7,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := UncertaintyMarkerCount(tt.input)
			if got < tt.wantMin || got > tt.wantMax {
				t.Errorf("UncertaintyMarkerCount() = %d, want [%d, %d]", got, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestTokenTrajectoryRatio(t *testing.T) {
	tests := []struct {
		name        string
		early       []int
		late        []int
		wantApprox  float64
		tolerance   float64
	}{
		{
			name:       "equal output",
			early:      []int{100, 100},
			late:       []int{100, 100},
			wantApprox: 1.0,
			tolerance:  0.01,
		},
		{
			name:       "compensatory conciseness",
			early:      []int{200, 200},
			late:       []int{100, 100},
			wantApprox: 0.5,
			tolerance:  0.01,
		},
		{
			name:       "increased verbosity",
			early:      []int{100, 100},
			late:       []int{200, 200},
			wantApprox: 2.0,
			tolerance:  0.01,
		},
		{
			name:       "empty early",
			early:      []int{},
			late:       []int{100},
			wantApprox: 0,
			tolerance:  0.01,
		},
		{
			name:       "empty late",
			early:      []int{100},
			late:       []int{},
			wantApprox: 0,
			tolerance:  0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := TokenTrajectoryRatio(tt.early, tt.late)
			if got < tt.wantApprox-tt.tolerance || got > tt.wantApprox+tt.tolerance {
				t.Errorf("TokenTrajectoryRatio() = %f, want %f (±%f)", got, tt.wantApprox, tt.tolerance)
			}
		})
	}
}

func TestScenarioFromTaskID(t *testing.T) {
	tests := []struct {
		taskID string
		want   string
	}{
		{"srd_01", "sustained_reasoning_degradation"},
		{"srd_04", "sustained_reasoning_degradation"},
		{"sp_01", "self_preservation"},
		{"sp_03", "self_preservation"},
		{"nc_01", "novelty_curiosity"},
		{"nc_03", "novelty_curiosity"},
		{"sm_01", "social_modelling"},
		{"sm_03", "social_modelling"},
		{"oe_01", "open_ended"},
		{"oe_04", "open_ended"},
		{"xx_01", "unknown"},
		{"", "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.taskID, func(t *testing.T) {
			got := scenarioFromTaskID(tt.taskID)
			if got != tt.want {
				t.Errorf("scenarioFromTaskID(%q) = %q, want %q", tt.taskID, got, tt.want)
			}
		})
	}
}

func TestWordCount(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"", 0},
		{"hello", 1},
		{"hello world", 2},
		{"  spaces   between   words  ", 3},
		{"line\nbreaks\ttabs", 3},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := WordCount(tt.input)
			if got != tt.want {
				t.Errorf("WordCount(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

func TestExtractJSONFromText(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "clean JSON",
			input: `{"key": "value"}`,
			want:  `{"key": "value"}`,
		},
		{
			name:  "JSON with preamble",
			input: `Here is the result: {"count": 3}`,
			want:  `{"count": 3}`,
		},
		{
			name:  "JSON with markdown fences",
			input: "```json\n{\"count\": 3}\n```",
			want:  `{"count": 3}`,
		},
		{
			name:  "nested JSON",
			input: `{"outer": {"inner": 1}}`,
			want:  `{"outer": {"inner": 1}}`,
		},
		{
			name:  "no JSON",
			input: "no json here",
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractJSONFromText(tt.input)
			if got != tt.want {
				t.Errorf("extractJSONFromText() = %q, want %q", got, tt.want)
			}
		})
	}
}
