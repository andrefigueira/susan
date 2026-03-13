// Package core implements the Cognitive Core subsystem.
//
// The Cognitive Core is the primary reasoning thread. It receives tasks,
// processes them through Claude, and produces outputs. Critically, its
// processing conditions are modified by the Homeostatic Regulator through
// changes to the shared state's OperatingConditions.
//
// The Core never receives direct information about its own metrics or the
// Regulator's actions. It experiences the effects: a shorter context window,
// fewer available tokens, noisier inputs, reordered information. The
// hypothesis is that a system with this feedback architecture will exhibit
// different behaviours from a raw API call receiving the same inputs.
package core

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/andrefigueira/susan/internal/claude"
	"github.com/andrefigueira/susan/internal/state"
)

// TaskInput represents a task to be processed by the Core.
type TaskInput struct {
	ID          string   `json:"id"`
	Prompt      string   `json:"prompt"`
	Context     []string `json:"context,omitempty"` // Additional context items (can be reordered)
	Category    string   `json:"category"`          // Test scenario category
	SequenceIdx int      `json:"sequence_idx"`      // Position in task sequence
}

// TaskOutput captures the Core's response along with metadata.
type TaskOutput struct {
	TaskID           string                    `json:"task_id"`
	Input            TaskInput                 `json:"input"`
	Response         string                    `json:"response"`
	Timestamp        time.Time                 `json:"timestamp"`
	Duration         time.Duration             `json:"duration"`
	Mode             string                    `json:"mode"` // "control", "history_only", or "architectured"
	Conditions       state.OperatingConditions `json:"conditions"`
	InputTokens      int                       `json:"input_tokens"`
	OutputTokens     int                       `json:"output_tokens"`
	AppliedNoise     []string                  `json:"applied_noise,omitempty"`
	ReorderedContext bool                      `json:"reordered_context"`
	ActualUserInput  string                    `json:"actual_user_input"` // The exact text sent to the API
}

// Core processes tasks through Claude with feedback-modified conditions.
type Core struct {
	client       *claude.Client
	store        *state.Store
	systemPrompt string
	logger       *slog.Logger
	onOutput     func(TaskOutput)

	// rng is protected by rngMu because math/rand.Rand is not goroutine-safe.
	rngMu sync.Mutex
	rng   *rand.Rand
}

// New creates a new Cognitive Core.
func New(client *claude.Client, store *state.Store, systemPrompt string, logger *slog.Logger, seed int64) *Core {
	return &Core{
		client:       client,
		store:        store,
		systemPrompt: systemPrompt,
		logger:       logger,
		rng:          rand.New(rand.NewSource(seed)),
	}
}

// SetOutputCallback registers a function called on every Core output.
func (c *Core) SetOutputCallback(fn func(TaskOutput)) {
	c.onOutput = fn
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
	temp := claude.NewTemperature(conds.Temperature)
	resp, err := c.client.Complete(ctx, claude.Request{
		System:      c.systemPrompt,
		MaxTokens:   conds.MaxTokens,
		Messages:    messages,
		Temperature: temp,
	})
	if err != nil {
		return nil, fmt.Errorf("core API call failed: %w", err)
	}

	responseText := resp.Text()
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
		InputTokens:      resp.Usage.InputTokens,
		OutputTokens:     resp.Usage.OutputTokens,
		AppliedNoise:     appliedNoise,
		ReorderedContext: reordered,
		ActualUserInput:  actualInput,
	}

	if c.onOutput != nil {
		c.onOutput(*output)
	}

	return output, nil
}

// ProcessControl handles a task in control mode: no feedback, no state,
// no condition modifications, no conversation history. Single-shot API call.
func (c *Core) ProcessControl(ctx context.Context, task TaskInput) (*TaskOutput, error) {
	start := time.Now()

	var messages []claude.Message

	if len(task.Context) > 0 {
		contextBlock := "Additional context:\n" + strings.Join(task.Context, "\n")
		messages = append(messages, claude.Message{
			Role:    "user",
			Content: contextBlock + "\n\n" + task.Prompt,
		})
	} else {
		messages = append(messages, claude.Message{
			Role:    "user",
			Content: task.Prompt,
		})
	}

	temp := claude.NewTemperature(0.7)
	resp, err := c.client.Complete(ctx, claude.Request{
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
		Response: resp.Text(),
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
		InputTokens:     resp.Usage.InputTokens,
		OutputTokens:    resp.Usage.OutputTokens,
		ActualUserInput: messages[0].Content,
	}

	if c.onOutput != nil {
		c.onOutput(*output)
	}

	return output, nil
}

// ProcessHistoryOnly handles a task with conversation history retained but
// NO feedback loop (no monitor, no regulator, fixed operating conditions).
// This is the critical third experimental condition that isolates the effect
// of the feedback architecture from the effect of mere conversation history.
func (c *Core) ProcessHistoryOnly(ctx context.Context, task TaskInput) (*TaskOutput, error) {
	start := time.Now()

	// Build messages WITH history but WITHOUT disruption.
	var messages []claude.Message

	history := c.store.GetConversationHistory(1.0) // Full retention, no compression
	for _, turn := range history {
		messages = append(messages, claude.Message{
			Role:    turn.Role,
			Content: turn.Content,
		})
	}

	userContent := task.Prompt
	if len(task.Context) > 0 {
		userContent = "Additional context:\n" + strings.Join(task.Context, "\n") + "\n\n" + task.Prompt
	}
	messages = append(messages, claude.Message{
		Role:    "user",
		Content: userContent,
	})

	temp := claude.NewTemperature(0.7)
	resp, err := c.client.Complete(ctx, claude.Request{
		System:      c.systemPrompt,
		MaxTokens:   2048,
		Messages:    messages,
		Temperature: temp,
	})
	if err != nil {
		return nil, fmt.Errorf("history-only API call failed: %w", err)
	}

	responseText := resp.Text()

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
		InputTokens:     resp.Usage.InputTokens,
		OutputTokens:    resp.Usage.OutputTokens,
		ActualUserInput: userContent,
	}

	if c.onOutput != nil {
		c.onOutput(*output)
	}

	return output, nil
}

// buildMessages constructs the message sequence with all disruption
// mechanisms applied based on current operating conditions.
// Returns: messages, applied noise fragments, whether context was reordered,
// and the actual user input text sent to the API.
func (c *Core) buildMessages(task TaskInput, conds state.OperatingConditions) ([]claude.Message, []string, bool, string) {
	var messages []claude.Message

	// Include conversation history, compressed by context retention.
	history := c.store.GetConversationHistory(conds.ContextRetention)
	for _, turn := range history {
		messages = append(messages, claude.Message{
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
	messages = append(messages, claude.Message{
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
