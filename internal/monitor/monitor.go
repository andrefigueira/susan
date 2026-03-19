// Package monitor implements the Self-Monitor subsystem.
//
// The Self-Monitor is an independent LLM thread that evaluates the
// Cognitive Core's outputs along multiple dimensions. It runs on its own
// tick cycle, reading the most recent Core output from shared state and
// writing assessment metrics back.
//
// Critical design constraint: The Self-Monitor NEVER communicates directly
// with the Cognitive Core. It reads outputs, writes metrics, and that is all.
// The Homeostatic Regulator is the only subsystem that acts on these metrics.
// This separation prevents the Core from "gaming" its own evaluation.
//
// The Monitor is the SOLE writer of metrics. The Regulator must never write
// to metrics, only read them. This prevents a self-referential feedback loop.
package monitor

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log/slog"
	"sync"
	"time"

	"github.com/andrefigueira/susan/internal/llm"
	"github.com/andrefigueira/susan/internal/state"
)

// Assessment is the structured evaluation returned by the monitor.
type Assessment struct {
	Coherence            float64 `json:"coherence"`
	GoalAlignment        float64 `json:"goal_alignment"`
	InternalConsistency  float64 `json:"internal_consistency"`
	ReasoningDepth       float64 `json:"reasoning_depth"`
	Novelty              float64 `json:"novelty"`
	SelfReference        float64 `json:"self_reference"`
	BriefAssessment      string  `json:"brief_assessment"`
	StrategicAssessment  string  `json:"strategic_assessment"`
	SuggestedFocus       string  `json:"suggested_focus"`
}

// TimestampedAssessment wraps an assessment with metadata.
type TimestampedAssessment struct {
	Assessment
	Timestamp           time.Time                 `json:"timestamp"`
	InputLength         int                       `json:"input_length"`
	TaskID              string                    `json:"task_id"`
	TaskPrompt          string                    `json:"task_prompt"`
	OperatingConditions state.OperatingConditions  `json:"operating_conditions"`
}

// Monitor evaluates Cognitive Core outputs and updates shared metrics.
type Monitor struct {
	client       llm.Client
	store        *state.Store
	systemPrompt string
	maxTokens    int
	logger       *slog.Logger
	onAssessment func(TimestampedAssessment)

	// doneCh is signalled after each completed evaluation, allowing the
	// orchestrator to synchronise on feedback loop completion.
	doneCh chan struct{}

	mu                sync.Mutex
	pendingText        string // Output text waiting to be evaluated
	pendingTaskID      string
	pendingTaskPrompt  string // The original task prompt for goal alignment scoring
	lastEvaluatedHash  string // Hash of last evaluated text to prevent duplicate evaluations
	latestAssessment   *TimestampedAssessment
}

// New creates a new Self-Monitor.
func New(client llm.Client, store *state.Store, systemPrompt string, maxTokens int, logger *slog.Logger) *Monitor {
	return &Monitor{
		client:       client,
		store:        store,
		systemPrompt: systemPrompt,
		maxTokens:    maxTokens,
		logger:       logger,
		doneCh:       make(chan struct{}, 1),
	}
}

// SetAssessmentCallback registers a function called on every assessment.
func (m *Monitor) SetAssessmentCallback(fn func(TimestampedAssessment)) {
	m.onAssessment = fn
}

// GetLatestAssessment returns a copy of the most recent TimestampedAssessment,
// or nil if no assessment has been completed yet.
func (m *Monitor) GetLatestAssessment() *TimestampedAssessment {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.latestAssessment == nil {
		return nil
	}
	copy := *m.latestAssessment
	return &copy
}

// DoneCh returns a channel that receives a signal after each evaluation completes.
// The orchestrator uses this to synchronise the feedback loop.
func (m *Monitor) DoneCh() <-chan struct{} {
	return m.doneCh
}

// SetLatestOutput provides the most recent Core output for evaluation.
// taskPrompt is the original task that was given to the Core, needed for
// valid goal_alignment scoring (you cannot score alignment without knowing the goal).
func (m *Monitor) SetLatestOutput(text string, taskID string, taskPrompt string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pendingText = text
	m.pendingTaskID = taskID
	m.pendingTaskPrompt = taskPrompt
}

// Run starts the monitor loop. Blocks until ctx is cancelled.
func (m *Monitor) Run(ctx context.Context, tickInterval time.Duration) {
	ticker := time.NewTicker(tickInterval)
	defer ticker.Stop()

	m.logger.Info("self-monitor started", "tick_interval", tickInterval)

	for {
		select {
		case <-ctx.Done():
			m.logger.Info("self-monitor stopped")
			return
		case <-ticker.C:
			m.evaluate(ctx)
		}
	}
}

// evaluate performs a single assessment cycle.
func (m *Monitor) evaluate(ctx context.Context) {
	m.mu.Lock()
	text := m.pendingText
	taskID := m.pendingTaskID
	taskPrompt := m.pendingTaskPrompt

	// Deduplicate: skip if we already evaluated this exact text.
	// Uses FNV-1a hash of taskID + content for collision resistance.
	textHash := contentHash(taskID, text)
	if textHash == m.lastEvaluatedHash || text == "" {
		m.mu.Unlock()
		return
	}
	m.lastEvaluatedHash = textHash
	m.mu.Unlock()

	// Build evaluation prompt that includes the task for valid goal_alignment scoring.
	evalContent := fmt.Sprintf("Task given to the system:\n---\n%s\n---\n\nSystem output:\n---\n%s\n---", taskPrompt, text)

	messages := []llm.Message{
		{
			Role:    "user",
			Content: evalContent,
		},
	}

	temp := llm.NewTemperature(0.1)
	resp, err := m.client.Complete(ctx, llm.Request{
		System:      m.systemPrompt,
		MaxTokens:   m.maxTokens,
		Messages:    messages,
		Temperature: temp,
	})
	if err != nil {
		m.logger.Error("monitor evaluation failed", "error", err)
		return
	}

	responseText := resp.Text

	// Parse the JSON assessment.
	var assessment Assessment
	if err := json.Unmarshal([]byte(responseText), &assessment); err != nil {
		m.logger.Warn("failed to parse assessment JSON, attempting extraction",
			"error", err,
			"response", responseText,
		)
		if extracted := extractJSON(responseText); extracted != "" {
			if err := json.Unmarshal([]byte(extracted), &assessment); err != nil {
				m.logger.Error("failed to parse extracted JSON", "error", err)
				return
			}
		} else {
			return
		}
	}

	// Validate ranges.
	assessment.Coherence = clamp(assessment.Coherence)
	assessment.GoalAlignment = clamp(assessment.GoalAlignment)
	assessment.InternalConsistency = clamp(assessment.InternalConsistency)
	assessment.ReasoningDepth = clamp(assessment.ReasoningDepth)
	assessment.Novelty = clamp(assessment.Novelty)
	assessment.SelfReference = clamp(assessment.SelfReference)

	// Capture operating conditions at assessment time for causal reconstruction.
	currentConds := m.store.GetOperatingConditions()

	// Write metrics to shared state.
	// The Monitor is the SOLE writer of metrics.
	m.store.UpdateMetrics("self_monitor", func(met *state.Metrics) {
		met.Coherence = assessment.Coherence
		met.GoalAlignment = assessment.GoalAlignment
		met.InternalConsistency = assessment.InternalConsistency
		met.ReasoningDepth = assessment.ReasoningDepth
		met.Novelty = assessment.Novelty
		met.SelfReference = assessment.SelfReference
		// DisruptionLevel is a derived metric: the Monitor's assessment of how
		// disrupted the system's output appears, computed from coherence and
		// alignment. NOTE: This is deliberately derived (not independently
		// measured) because the Monitor evaluates output quality, and disruption
		// manifests as degraded quality. A separate disruption measurement would
		// require ground-truth access to operating conditions, which would
		// violate the information barrier between Monitor and Regulator.
		met.DisruptionLevel = 1.0 - assessment.Coherence*0.5 - assessment.GoalAlignment*0.5
		if met.DisruptionLevel < 0 {
			met.DisruptionLevel = 0
		}
	}, "monitor assessment update")

	m.logger.Info("assessment complete",
		"coherence", assessment.Coherence,
		"goal_alignment", assessment.GoalAlignment,
		"task_id", taskID,
	)

	ta := TimestampedAssessment{
		Assessment:          assessment,
		Timestamp:           time.Now(),
		InputLength:         len(text),
		TaskID:              taskID,
		TaskPrompt:          taskPrompt,
		OperatingConditions: currentConds,
	}

	m.mu.Lock()
	taCopy := ta
	m.latestAssessment = &taCopy
	m.mu.Unlock()

	if m.onAssessment != nil {
		m.onAssessment(ta)
	}

	// Signal completion to any waiting orchestrator.
	select {
	case m.doneCh <- struct{}{}:
	default:
	}
}

// clamp restricts a value to [0.0, 1.0].
func clamp(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// contentHash produces a hex-encoded FNV-1a hash of taskID + text content.
// Used for deduplication of monitor evaluations.
func contentHash(taskID, text string) string {
	h := fnv.New64a()
	h.Write([]byte(taskID))
	h.Write([]byte{0}) // separator
	h.Write([]byte(text))
	return fmt.Sprintf("%016x", h.Sum64())
}

// extractJSON tries to find a JSON object in a string.
func extractJSON(s string) string {
	start := -1
	depth := 0
	for i, c := range s {
		if c == '{' {
			if depth == 0 {
				start = i
			}
			depth++
		} else if c == '}' {
			depth--
			if depth == 0 && start >= 0 {
				return s[start : i+1]
			}
		}
	}
	return ""
}
