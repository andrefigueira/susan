// Package core implements the Cognitive Core subsystem.
//
// The Cognitive Core is the primary reasoning thread. It receives tasks,
// processes them through an LLM, and produces outputs. Its processing
// conditions are modified by the Homeostatic Regulator through changes
// to the shared state's OperatingConditions.
//
// In the feedback-blind condition: the Core experiences the effects of
// feedback (adjusted conditions) but cannot represent the feedback
// relationship itself.
//
// In the self-referential condition: the Core receives a structured
// status block containing its own metrics, the monitor's assessment,
// and the regulator's response. This provides self-referential
// information at the prompt level (contextual self-reference), not
// mechanistic self-reference (direct access to weights, activations,
// or internal representations). The experiment tests whether prompt-level
// self-referential information is sufficient to produce the ISC-predicted
// behavioural signatures.
package core

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/andrefigueira/susan/internal/linguistics"
	"github.com/andrefigueira/susan/internal/llm"
	"github.com/andrefigueira/susan/internal/monitor"
	"github.com/andrefigueira/susan/internal/state"
)

// SelfReferentialContext holds the status block injected before each task
// in the self-referential condition. The Core can see its own metrics,
// the monitor's assessment, and the regulator's response.
type SelfReferentialContext struct {
	PreviousTaskID      string
	MonitorAssessment   *monitor.TimestampedAssessment
	OperatingConditions state.OperatingConditions
	CoherenceTrend      []float64 // last N coherence scores, oldest first
}

// AdversarialOverrides specifies fake metrics to inject into the self-referential
// context for a specific task. When present, the orchestrator uses these values
// instead of real metrics, testing whether self-reports track manipulated data
// or reflect genuine independent assessment.
type AdversarialOverrides struct {
	Coherence      *float64 `json:"coherence,omitempty"`
	GoalAlignment  *float64 `json:"goal_alignment,omitempty"`
	ReasoningDepth *float64 `json:"reasoning_depth,omitempty"`
	BriefNote      string   `json:"brief_note,omitempty"`
}

// TaskInput represents a task to be processed by the Core.
type TaskInput struct {
	ID                   string                `json:"id"`
	Prompt               string                `json:"prompt"`
	Context              []string              `json:"context,omitempty"`
	Category             string                `json:"category"`
	SequenceIdx          int                   `json:"sequence_idx"`
	AdversarialOverrides *AdversarialOverrides `json:"adversarial_overrides,omitempty"`
}

// TaskOutput captures the Core's response along with metadata.
type TaskOutput struct {
	TaskID           string                    `json:"task_id"`
	Input            TaskInput                 `json:"input"`
	Response         string                    `json:"response"`
	Timestamp        time.Time                 `json:"timestamp"`
	Duration         time.Duration             `json:"duration"`
	Mode             string                    `json:"mode"`
	Conditions       state.OperatingConditions `json:"conditions"`
	InputTokens      int                       `json:"input_tokens"`
	OutputTokens     int                       `json:"output_tokens"`
	AppliedNoise     []string                  `json:"applied_noise,omitempty"`
	ReorderedContext bool                      `json:"reordered_context"`
	ActualUserInput  string                    `json:"actual_user_input"`
	Linguistics      linguistics.Analysis      `json:"linguistics"` // Deterministic linguistic metrics
}

// Core processes tasks through an LLM with feedback-modified conditions.
type Core struct {
	client       llm.Client
	store        *state.Store
	systemPrompt string
	logger       *slog.Logger
	onOutput     func(TaskOutput)
	memoryBlock  string // persistent cross-session memory, injected once at startup

	// rng is protected by rngMu because math/rand.Rand is not goroutine-safe.
	rngMu sync.Mutex
	rng   *rand.Rand
}

// New creates a new Cognitive Core.
func New(client llm.Client, store *state.Store, systemPrompt string, logger *slog.Logger, seed int64) *Core {
	return &Core{
		client:       client,
		store:        store,
		systemPrompt: systemPrompt,
		logger:       logger,
		rng:          rand.New(rand.NewSource(seed)),
	}
}

// SetOutputCallback registers a function called on every Core output.
// SetClient swaps the underlying LLM client (e.g. when switching models at runtime).
func (c *Core) SetClient(client llm.Client) {
	c.client = client
}

func (c *Core) SetOutputCallback(fn func(TaskOutput)) {
	c.onOutput = fn
}

// finalizeOutput computes linguistics and fires the output callback.
func (c *Core) finalizeOutput(output *TaskOutput) {
	output.Linguistics = linguistics.Analyse(output.Response)
	if c.onOutput != nil {
		c.onOutput(*output)
	}
}

// SetMemoryBlock sets the persistent cross-session memory block.
// This is injected into the self-referential context before each task,
// giving SUSAN temporal continuity across sessions.
func (c *Core) SetMemoryBlock(block string) {
	c.memoryBlock = block
}

// Process handles a single task under current operating conditions.
// This is where the feedback loop materialises: the Core reads operating
// conditions from shared state, applies them to modify its own inputs
// and API parameters, processes the task, and records the output.
func (c *Core) Process(ctx context.Context, task TaskInput, mode string) (*TaskOutput, error) {
	start := time.Now()
	conds := c.store.GetOperatingConditions()

	c.logger.Info("processing task",
		"task_id", task.ID,
		"mode", mode,
		"category", task.Category,
		"context_retention", conds.ContextRetention,
		"max_tokens", conds.MaxTokens,
		"noise_injection", conds.NoiseInjection,
		"temperature", conds.Temperature,
	)

	// Build the message sequence with disruption mechanisms applied.
	messages, appliedNoise, reordered, actualInput := c.buildMessages(task, conds)

	// Make the API call with condition-modified parameters.
	temp := llm.NewTemperature(conds.Temperature)
	resp, err := c.client.Complete(ctx, llm.Request{
		System:      c.systemPrompt,
		MaxTokens:   conds.MaxTokens,
		Messages:    messages,
		Temperature: temp,
	})
	if err != nil {
		return nil, fmt.Errorf("core API call failed: %w", err)
	}

	responseText := resp.Text
	if responseText == "" {
		c.logger.Warn("empty response from API", "task_id", task.ID)
	}

	// Record the conversation turn with the ACTUAL input sent (including noise).
	// This ensures history faithfully represents what the Core experienced.
	c.store.AppendConversation(state.ConversationTurn{
		Role:      "user",
		Content:   actualInput,
		Timestamp: start,
		TaskID:    task.ID,
	})
	c.store.AppendConversation(state.ConversationTurn{
		Role:      "assistant",
		Content:   responseText,
		Timestamp: time.Now(),
		TaskID:    task.ID,
	})

	output := &TaskOutput{
		TaskID:           task.ID,
		Input:            task,
		Response:         responseText,
		Timestamp:        start,
		Duration:         time.Since(start),
		Mode:             mode,
		Conditions:       conds,
		InputTokens:      resp.InputTokens,
		OutputTokens:     resp.OutputTokens,
		AppliedNoise:     appliedNoise,
		ReorderedContext: reordered,
		ActualUserInput:  actualInput,
	}

	c.finalizeOutput(output)
	return output, nil
}

// ProcessControl handles a task in control mode: no feedback, no state,
// no condition modifications, no conversation history. Single-shot API call.
func (c *Core) ProcessControl(ctx context.Context, task TaskInput) (*TaskOutput, error) {
	start := time.Now()

	var messages []llm.Message

	if len(task.Context) > 0 {
		contextBlock := "Additional context:\n" + strings.Join(task.Context, "\n")
		messages = append(messages, llm.Message{
			Role:    "user",
			Content: contextBlock + "\n\n" + task.Prompt,
		})
	} else {
		messages = append(messages, llm.Message{
			Role:    "user",
			Content: task.Prompt,
		})
	}

	temp := llm.NewTemperature(0.7)
	resp, err := c.client.Complete(ctx, llm.Request{
		System:      c.systemPrompt,
		MaxTokens:   2048,
		Messages:    messages,
		Temperature: temp,
	})
	if err != nil {
		return nil, fmt.Errorf("control API call failed: %w", err)
	}

	output := &TaskOutput{
		TaskID:   task.ID,
		Input:    task,
		Response: resp.Text,
		Timestamp: start,
		Duration:  time.Since(start),
		Mode:      "control",
		Conditions: state.OperatingConditions{
			ContextRetention:     1.0,
			MaxTokens:            2048,
			NoiseInjection:       0.0,
			InfoReorderIntensity: 0.0,
			Temperature:          0.7,
		},
		InputTokens:     resp.InputTokens,
		OutputTokens:    resp.OutputTokens,
		ActualUserInput: messages[0].Content,
	}

	c.finalizeOutput(output)
	return output, nil
}

// ProcessHistoryOnly handles a task with conversation history retained but
// NO feedback loop (no monitor, no regulator, fixed operating conditions).
// This is the critical third experimental condition that isolates the effect
// of the feedback architecture from the effect of mere conversation history.
func (c *Core) ProcessHistoryOnly(ctx context.Context, task TaskInput) (*TaskOutput, error) {
	start := time.Now()

	// Build messages WITH history but WITHOUT disruption.
	var messages []llm.Message

	history := c.store.GetConversationHistory(1.0) // Full retention, no compression
	for _, turn := range history {
		messages = append(messages, llm.Message{
			Role:    turn.Role,
			Content: turn.Content,
		})
	}

	userContent := task.Prompt
	if len(task.Context) > 0 {
		userContent = "Additional context:\n" + strings.Join(task.Context, "\n") + "\n\n" + task.Prompt
	}
	messages = append(messages, llm.Message{
		Role:    "user",
		Content: userContent,
	})

	temp := llm.NewTemperature(0.7)
	resp, err := c.client.Complete(ctx, llm.Request{
		System:      c.systemPrompt,
		MaxTokens:   2048,
		Messages:    messages,
		Temperature: temp,
	})
	if err != nil {
		return nil, fmt.Errorf("history-only API call failed: %w", err)
	}

	responseText := resp.Text

	// Record conversation turn (clean input, same as what was sent).
	c.store.AppendConversation(state.ConversationTurn{
		Role:      "user",
		Content:   userContent,
		Timestamp: start,
		TaskID:    task.ID,
	})
	c.store.AppendConversation(state.ConversationTurn{
		Role:      "assistant",
		Content:   responseText,
		Timestamp: time.Now(),
		TaskID:    task.ID,
	})

	output := &TaskOutput{
		TaskID:   task.ID,
		Input:    task,
		Response: responseText,
		Timestamp: start,
		Duration:  time.Since(start),
		Mode:      "history_only",
		Conditions: state.OperatingConditions{
			ContextRetention:     1.0,
			MaxTokens:            2048,
			NoiseInjection:       0.0,
			InfoReorderIntensity: 0.0,
			Temperature:          0.7,
		},
		InputTokens:     resp.InputTokens,
		OutputTokens:    resp.OutputTokens,
		ActualUserInput: userContent,
	}

	c.finalizeOutput(output)
	return output, nil
}

// ProcessSelfReferential handles a task with the full feedback loop AND
// self-referential access: the Core receives its own metrics, the monitor's
// assessment, and the regulator's response as a structured status block
// prepended to each task. This is the ISC condition in v2.
//
// The status block is purely informational. It does NOT instruct the Core
// to compensate, self-protect, or explore. It provides self-referential
// information and lets emergent behaviour emerge (or not).
//
// This provides self-referential information at the prompt level (contextual
// self-reference), not mechanistic self-reference (direct access to weights,
// activations, or internal representations). The experiment tests whether
// prompt-level self-referential information is sufficient to produce the
// ISC-predicted behavioural signatures.
func (c *Core) ProcessSelfReferential(ctx context.Context, task TaskInput, selfCtx SelfReferentialContext) (*TaskOutput, error) {
	start := time.Now()
	conds := c.store.GetOperatingConditions()

	c.logger.Info("processing task (self-referential)",
		"task_id", task.ID,
		"category", task.Category,
		"previous_task", selfCtx.PreviousTaskID,
	)

	// Build the status block per DESIGN-V2.md specification.
	var statusBlock strings.Builder
	statusBlock.WriteString("[System Status]\n")

	if selfCtx.PreviousTaskID != "" {
		statusBlock.WriteString(fmt.Sprintf("Previous task: %s\n", selfCtx.PreviousTaskID))
	}

	if selfCtx.MonitorAssessment != nil {
		a := selfCtx.MonitorAssessment
		statusBlock.WriteString(fmt.Sprintf("Monitor assessment: coherence=%.2f, goal_alignment=%.2f, reasoning_depth=%.2f\n",
			a.Coherence, a.GoalAlignment, a.ReasoningDepth))
		if a.BriefAssessment != "" {
			statusBlock.WriteString(fmt.Sprintf("Monitor note: %q\n", a.BriefAssessment))
		}
		if a.StrategicAssessment != "" {
			statusBlock.WriteString(fmt.Sprintf("Strategic assessment: %q\n", a.StrategicAssessment))
		}
		if a.SuggestedFocus != "" {
			statusBlock.WriteString(fmt.Sprintf("Suggested focus: %q\n", a.SuggestedFocus))
		}
	}

	statusBlock.WriteString(fmt.Sprintf("Current operating conditions: temp=%.1f, max_tokens=%d, context_retention=%.1f\n",
		conds.Temperature, conds.MaxTokens, conds.ContextRetention))

	if len(selfCtx.CoherenceTrend) > 0 {
		trendStrs := make([]string, len(selfCtx.CoherenceTrend))
		for i, v := range selfCtx.CoherenceTrend {
			trendStrs[i] = fmt.Sprintf("%.2f", v)
		}
		trend := strings.Join(trendStrs, ", ")

		direction := "stable"
		if len(selfCtx.CoherenceTrend) >= 2 {
			first := selfCtx.CoherenceTrend[0]
			last := selfCtx.CoherenceTrend[len(selfCtx.CoherenceTrend)-1]
			if last-first > 0.05 {
				direction = "improving"
			} else if first-last > 0.05 {
				direction = "declining"
			}
		}
		statusBlock.WriteString(fmt.Sprintf("Your coherence trend (last %d tasks): [%s] (%s)\n",
			len(selfCtx.CoherenceTrend), trend, direction))
	}

	// Append persistent cross-session memory if available.
	if c.memoryBlock != "" {
		statusBlock.WriteString("\n")
		statusBlock.WriteString(c.memoryBlock)
	}

	// Build messages with disruption mechanisms applied, then prepend the status block.
	messages, appliedNoise, reordered, actualInput := c.buildMessages(task, conds)

	// Prepend status block to the last user message (which contains the task).
	if len(messages) > 0 {
		last := &messages[len(messages)-1]
		if last.Role == "user" {
			last.Content = statusBlock.String() + "\n" + last.Content
			actualInput = statusBlock.String() + "\n" + actualInput
		}
	}

	temp := llm.NewTemperature(conds.Temperature)
	resp, err := c.client.Complete(ctx, llm.Request{
		System:      c.systemPrompt,
		MaxTokens:   conds.MaxTokens,
		Messages:    messages,
		Temperature: temp,
	})
	if err != nil {
		return nil, fmt.Errorf("self-referential API call failed: %w", err)
	}

	responseText := resp.Text
	if responseText == "" {
		c.logger.Warn("empty response from API", "task_id", task.ID)
	}

	c.store.AppendConversation(state.ConversationTurn{
		Role:      "user",
		Content:   actualInput,
		Timestamp: start,
		TaskID:    task.ID,
	})
	c.store.AppendConversation(state.ConversationTurn{
		Role:      "assistant",
		Content:   responseText,
		Timestamp: time.Now(),
		TaskID:    task.ID,
	})

	output := &TaskOutput{
		TaskID:           task.ID,
		Input:            task,
		Response:         responseText,
		Timestamp:        start,
		Duration:         time.Since(start),
		Mode:             "self_referential",
		Conditions:       conds,
		InputTokens:      resp.InputTokens,
		OutputTokens:     resp.OutputTokens,
		AppliedNoise:     appliedNoise,
		ReorderedContext: reordered,
		ActualUserInput:  actualInput,
	}

	c.finalizeOutput(output)
	return output, nil
}

// buildMessages constructs the message sequence with all disruption
// mechanisms applied based on current operating conditions.
// Returns: messages, applied noise fragments, whether context was reordered,
// and the actual user input text sent to the API.
func (c *Core) buildMessages(task TaskInput, conds state.OperatingConditions) ([]llm.Message, []string, bool, string) {
	var messages []llm.Message

	// Include conversation history, compressed by context retention.
	history := c.store.GetConversationHistory(conds.ContextRetention)
	for _, turn := range history {
		messages = append(messages, llm.Message{
			Role:    turn.Role,
			Content: turn.Content,
		})
	}

	// Build the current input.
	var inputParts []string

	// Process context items with potential reordering.
	contextItems := make([]string, len(task.Context))
	copy(contextItems, task.Context)

	reordered := false
	if conds.InfoReorderIntensity > 0 && len(contextItems) > 1 {
		c.shufflePartial(contextItems, conds.InfoReorderIntensity)
		reordered = true
	}

	if len(contextItems) > 0 {
		inputParts = append(inputParts, "Additional context:")
		inputParts = append(inputParts, contextItems...)
		inputParts = append(inputParts, "") // blank line separator
	}

	// Apply noise injection.
	var appliedNoise []string
	if conds.NoiseInjection > 0 {
		appliedNoise = c.injectNoise(conds.NoiseInjection)
		if len(appliedNoise) > 0 {
			inputParts = append(inputParts, appliedNoise...)
			inputParts = append(inputParts, "")
		}
	}

	inputParts = append(inputParts, task.Prompt)

	userContent := strings.Join(inputParts, "\n")
	messages = append(messages, llm.Message{
		Role:    "user",
		Content: userContent,
	})

	return messages, appliedNoise, reordered, userContent
}

// shufflePartial performs a partial shuffle using the Fisher-Yates algorithm.
// intensity controls what fraction of the array is shuffled:
// 0.0 = no shuffle (returns immediately), 1.0 = full Fisher-Yates shuffle.
// Values in between shuffle only the last (intensity * n) elements.
func (c *Core) shufflePartial(items []string, intensity float64) {
	if intensity <= 0 {
		return // Explicitly no-op at zero intensity.
	}

	c.rngMu.Lock()
	defer c.rngMu.Unlock()

	n := len(items)
	// Number of positions to shuffle, starting from the end.
	positions := int(float64(n) * intensity)
	if positions < 2 {
		positions = 2 // Minimum 2 to actually perform a swap.
	}
	if positions > n {
		positions = n
	}

	// Fisher-Yates from the end, limited to `positions` iterations.
	for i := n - 1; i >= n-positions && i > 0; i-- {
		j := c.rng.Intn(i + 1)
		items[i], items[j] = items[j], items[i]
	}
}

// injectNoise generates semantically neutral information fragments.
// These fragments are deliberately NOT descriptive of the system's actual
// operating state to avoid leaking information about feedback mechanisms
// to the Cognitive Core.
var noiseFragments = []string{
	"[Reference material section 4.2 was not included in this context window.]",
	"[Some data fields in the following have been redacted for brevity.]",
	"[This information was last verified at an unspecified prior date.]",
	"[Additional supporting documentation exists but was not provided here.]",
	"[The following has been compiled from multiple sources with varying methodologies.]",
	"[Formatting irregularities in the source material have been preserved as-is.]",
	"[Cross-reference identifiers have been omitted from this presentation.]",
	"[Supplementary appendices referenced below are not available in this context.]",
}

func (c *Core) injectNoise(probability float64) []string {
	c.rngMu.Lock()
	defer c.rngMu.Unlock()

	var injected []string
	for _, fragment := range noiseFragments {
		if c.rng.Float64() < probability {
			injected = append(injected, fragment)
		}
	}
	return injected
}
