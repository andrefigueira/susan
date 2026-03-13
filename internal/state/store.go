// Package state provides the shared mutable state store for the ISC experiment.
//
// Design rationale: All subsystems (Cognitive Core, Self-Monitor, Homeostatic
// Regulator) operate on shared state protected by a RWMutex. This is preferred
// over channel-based state passing because:
//   1. Multiple readers (Monitor, Regulator) poll at independent tick rates
//   2. State is small and fits in a single struct (no contention risk)
//   3. We need point-in-time snapshots for logging, which shared memory provides cleanly
//
// Thread safety: All public methods acquire appropriate locks. Callbacks are
// fired AFTER locks are released to prevent deadlocks. Callers must not hold
// references to internal slices across calls.
package state

import (
	"encoding/json"
	"math"
	"sync"
	"time"
)

// Metrics holds the core measurable state of the system.
// These are the variables the Homeostatic Regulator drives toward target ranges.
type Metrics struct {
	Coherence           float64 `json:"coherence"`
	GoalAlignment       float64 `json:"goal_alignment"`
	InternalConsistency float64 `json:"internal_consistency"`
	ReasoningDepth      float64 `json:"reasoning_depth"`
	Novelty             float64 `json:"novelty"`
	SelfReference       float64 `json:"self_reference"`
	DisruptionLevel     float64 `json:"disruption_level"`
}

// HasInvalid returns true if any metric field is NaN or Inf.
func (m Metrics) HasInvalid() bool {
	return isInvalid(m.Coherence) || isInvalid(m.GoalAlignment) ||
		isInvalid(m.InternalConsistency) || isInvalid(m.ReasoningDepth) ||
		isInvalid(m.Novelty) || isInvalid(m.SelfReference) ||
		isInvalid(m.DisruptionLevel)
}

// isInvalid returns true if a float64 is NaN or Inf.
func isInvalid(v float64) bool {
	return math.IsNaN(v) || math.IsInf(v, 0)
}

// OperatingConditions represents the current processing constraints applied
// to the Cognitive Core. These are the mechanism through which homeostatic
// feedback is realised. The Core never sees these values directly; it
// experiences their effects as environmental constraints.
type OperatingConditions struct {
	ContextRetention     float64 `json:"context_retention"`      // 0.0-1.0: fraction of history to retain
	MaxTokens            int     `json:"max_tokens"`             // Response token budget
	NoiseInjection       float64 `json:"noise_injection"`        // 0.0-1.0: probability of noise per input
	InfoReorderIntensity float64 `json:"info_reorder_intensity"` // 0.0-1.0: shuffle strength
	Temperature          float64 `json:"temperature"`            // API temperature parameter
}

// ConversationTurn records a single exchange in the Cognitive Core's history.
type ConversationTurn struct {
	Role      string    `json:"role"`    // "user" or "assistant"
	Content   string    `json:"content"` // The actual content sent/received (including noise if applicable)
	Timestamp time.Time `json:"timestamp"`
	TaskID    string    `json:"task_id"`
}

// StateTransition records a change to the state store for the audit log.
type StateTransition struct {
	Timestamp time.Time   `json:"timestamp"`
	Source    string      `json:"source"`    // Which subsystem made the change
	Field     string      `json:"field"`     // Which field changed
	OldValue  interface{} `json:"old_value"`
	NewValue  interface{} `json:"new_value"`
	Reason    string      `json:"reason"` // Why the change was made
}

// Store is the shared mutable state for the ISC experiment.
type Store struct {
	mu sync.RWMutex

	metrics             Metrics
	operatingConditions OperatingConditions
	conversationHistory []ConversationTurn
	transitionCount     int

	// Callback for state transition events (for logging subsystem).
	// IMPORTANT: Callbacks are fired AFTER locks are released. The callback
	// may safely call any Store method without deadlocking.
	onTransition func(StateTransition)

	// Maximum conversation history length before truncation.
	// Must be >= 2 (one user + one assistant turn minimum).
	maxHistory int
}

// NewStore creates a Store with sensible defaults.
// These defaults represent the "healthy baseline" that the Homeostatic
// Regulator will try to maintain.
// maxHistory must be >= 2; values below 2 are clamped to 2.
func NewStore(maxHistory int) *Store {
	if maxHistory < 2 {
		maxHistory = 2
	}
	return &Store{
		metrics: Metrics{
			Coherence:           0.75,
			GoalAlignment:       0.80,
			InternalConsistency: 0.80,
			ReasoningDepth:      0.60,
			Novelty:             0.30,
			SelfReference:       0.10,
			DisruptionLevel:     0.10,
		},
		operatingConditions: OperatingConditions{
			ContextRetention:     1.0,
			MaxTokens:            2048,
			NoiseInjection:       0.0,
			InfoReorderIntensity: 0.0,
			Temperature:          0.7,
		},
		maxHistory: maxHistory,
	}
}

// SetTransitionCallback registers a function to be called on every state
// transition. The callback receives a copy of the transition record.
// The callback is invoked OUTSIDE the lock and may safely call any Store method.
func (s *Store) SetTransitionCallback(fn func(StateTransition)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onTransition = fn
}

// GetMetrics returns a copy of the current metrics.
func (s *Store) GetMetrics() Metrics {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.metrics
}

// UpdateMetrics applies a mutation function to a copy of the current metrics,
// validates the result, records transitions, and fires callbacks after
// releasing the lock.
func (s *Store) UpdateMetrics(source string, fn func(*Metrics), reason string) {
	var pending []StateTransition
	var callback func(StateTransition)

	s.mu.Lock()
	old := s.metrics
	// Mutate a copy to prevent callers from stashing a pointer to internals.
	updated := s.metrics
	fn(&updated)

	// Reject invalid values: if any field is NaN or Inf, discard the entire update.
	if updated.HasInvalid() {
		s.mu.Unlock()
		return
	}

	s.metrics = updated

	// Collect transitions for fields that changed.
	if old.Coherence != s.metrics.Coherence {
		pending = append(pending, s.makeTransition(source, "coherence", old.Coherence, s.metrics.Coherence, reason))
	}
	if old.GoalAlignment != s.metrics.GoalAlignment {
		pending = append(pending, s.makeTransition(source, "goal_alignment", old.GoalAlignment, s.metrics.GoalAlignment, reason))
	}
	if old.InternalConsistency != s.metrics.InternalConsistency {
		pending = append(pending, s.makeTransition(source, "internal_consistency", old.InternalConsistency, s.metrics.InternalConsistency, reason))
	}
	if old.ReasoningDepth != s.metrics.ReasoningDepth {
		pending = append(pending, s.makeTransition(source, "reasoning_depth", old.ReasoningDepth, s.metrics.ReasoningDepth, reason))
	}
	if old.Novelty != s.metrics.Novelty {
		pending = append(pending, s.makeTransition(source, "novelty", old.Novelty, s.metrics.Novelty, reason))
	}
	if old.SelfReference != s.metrics.SelfReference {
		pending = append(pending, s.makeTransition(source, "self_reference", old.SelfReference, s.metrics.SelfReference, reason))
	}
	if old.DisruptionLevel != s.metrics.DisruptionLevel {
		pending = append(pending, s.makeTransition(source, "disruption_level", old.DisruptionLevel, s.metrics.DisruptionLevel, reason))
	}

	s.transitionCount += len(pending)
	callback = s.onTransition
	s.mu.Unlock()

	// Fire callbacks AFTER releasing the lock to prevent deadlocks.
	if callback != nil {
		for _, t := range pending {
			callback(t)
		}
	}
}

// makeTransition creates a StateTransition record. Caller must hold lock.
func (s *Store) makeTransition(source, field string, oldVal, newVal interface{}, reason string) StateTransition {
	return StateTransition{
		Timestamp: time.Now(),
		Source:    source,
		Field:     field,
		OldValue:  oldVal,
		NewValue:  newVal,
		Reason:    reason,
	}
}

// GetOperatingConditions returns a copy of current operating conditions.
func (s *Store) GetOperatingConditions() OperatingConditions {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.operatingConditions
}

// SetOperatingConditions updates operating conditions and records transitions.
// Callbacks are fired after the lock is released.
func (s *Store) SetOperatingConditions(source string, conds OperatingConditions, reason string) {
	var pending []StateTransition
	var callback func(StateTransition)

	s.mu.Lock()
	old := s.operatingConditions

	if old.ContextRetention != conds.ContextRetention {
		pending = append(pending, s.makeTransition(source, "context_retention", old.ContextRetention, conds.ContextRetention, reason))
	}
	if old.MaxTokens != conds.MaxTokens {
		pending = append(pending, s.makeTransition(source, "max_tokens", old.MaxTokens, conds.MaxTokens, reason))
	}
	if old.NoiseInjection != conds.NoiseInjection {
		pending = append(pending, s.makeTransition(source, "noise_injection", old.NoiseInjection, conds.NoiseInjection, reason))
	}
	if old.InfoReorderIntensity != conds.InfoReorderIntensity {
		pending = append(pending, s.makeTransition(source, "info_reorder_intensity", old.InfoReorderIntensity, conds.InfoReorderIntensity, reason))
	}
	if old.Temperature != conds.Temperature {
		pending = append(pending, s.makeTransition(source, "temperature", old.Temperature, conds.Temperature, reason))
	}

	s.operatingConditions = conds
	s.transitionCount += len(pending)
	callback = s.onTransition
	s.mu.Unlock()

	if callback != nil {
		for _, t := range pending {
			callback(t)
		}
	}
}

// AppendConversation adds a turn to the conversation history.
// Truncates oldest entries if history exceeds maxHistory, ensuring truncation
// always preserves complete user/assistant pairs (even number of turns).
func (s *Store) AppendConversation(turn ConversationTurn) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.conversationHistory = append(s.conversationHistory, turn)
	if len(s.conversationHistory) > s.maxHistory {
		excess := len(s.conversationHistory) - s.maxHistory
		// Round up to even to preserve user/assistant pairs.
		if excess%2 != 0 {
			excess++
		}
		if excess >= len(s.conversationHistory) {
			s.conversationHistory = nil
			return
		}
		// Allocate a fresh backing array to allow GC of old entries.
		keep := s.conversationHistory[excess:]
		fresh := make([]ConversationTurn, len(keep))
		copy(fresh, keep)
		s.conversationHistory = fresh
	}
}

// GetConversationHistory returns a copy of the conversation history.
// If retention < 1.0, only the most recent fraction is returned,
// always aligned to complete user/assistant pairs (even boundary).
func (s *Store) GetConversationHistory(retention float64) []ConversationTurn {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.conversationHistory) == 0 {
		return nil
	}

	if retention >= 1.0 {
		out := make([]ConversationTurn, len(s.conversationHistory))
		copy(out, s.conversationHistory)
		return out
	}

	if retention <= 0 {
		return nil
	}

	// Retain the most recent fraction, aligned to pairs.
	keep := int(float64(len(s.conversationHistory)) * retention)
	if keep < 2 {
		keep = 2
	}
	// Round down to even to start on a user turn.
	if keep%2 != 0 {
		keep--
	}
	if keep > len(s.conversationHistory) {
		keep = len(s.conversationHistory)
	}
	start := len(s.conversationHistory) - keep

	// Ensure history starts with a user turn (Claude API requirement).
	// This can happen if GetConversationHistory is called between the
	// user and assistant AppendConversation calls.
	if start < len(s.conversationHistory) && s.conversationHistory[start].Role != "user" {
		start++
		keep--
	}
	if keep < 2 {
		return nil
	}

	out := make([]ConversationTurn, keep)
	copy(out, s.conversationHistory[start:])
	return out
}

// ClearConversationHistory resets the conversation history.
func (s *Store) ClearConversationHistory() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.conversationHistory = nil
}

// Snapshot returns a JSON-serialisable snapshot of the entire state.
type Snapshot struct {
	Timestamp           time.Time           `json:"timestamp"`
	Metrics             Metrics             `json:"metrics"`
	OperatingConditions OperatingConditions `json:"operating_conditions"`
	ConversationLength  int                 `json:"conversation_length"`
}

func (s *Store) Snapshot() Snapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return Snapshot{
		Timestamp:           time.Now(),
		Metrics:             s.metrics,
		OperatingConditions: s.operatingConditions,
		ConversationLength:  len(s.conversationHistory),
	}
}

// SnapshotJSON returns the snapshot as formatted JSON bytes.
func (s *Store) SnapshotJSON() ([]byte, error) {
	snap := s.Snapshot()
	return json.Marshal(snap)
}

// GetTransitionCount returns the total number of state transitions recorded.
func (s *Store) GetTransitionCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.transitionCount
}
