package state

import (
	"math"
	"sync"
	"testing"
)

func TestStore_MetricsReadWrite(t *testing.T) {
	s := NewStore(20)

	m := s.GetMetrics()
	if m.Coherence != 0.75 {
		t.Errorf("expected default coherence 0.75, got %f", m.Coherence)
	}

	s.UpdateMetrics("test", func(m *Metrics) {
		m.Coherence = 0.5
	}, "test update")

	m = s.GetMetrics()
	if m.Coherence != 0.5 {
		t.Errorf("expected coherence 0.5, got %f", m.Coherence)
	}
}

func TestStore_TransitionCallbackFiresAfterLockRelease(t *testing.T) {
	s := NewStore(20)

	var transitions []StateTransition
	s.SetTransitionCallback(func(tr StateTransition) {
		transitions = append(transitions, tr)
		// This would deadlock in the old implementation.
		// The callback should be able to safely call back into the store.
		_ = s.GetMetrics()
		_ = s.Snapshot()
	})

	s.UpdateMetrics("test", func(m *Metrics) {
		m.Coherence = 0.6
		m.GoalAlignment = 0.9
	}, "multi-field update")

	if len(transitions) != 2 {
		t.Errorf("expected 2 transitions, got %d", len(transitions))
	}

	if transitions[0].Field != "coherence" {
		t.Errorf("expected first transition field 'coherence', got %q", transitions[0].Field)
	}
}

func TestStore_SetOperatingConditions_TransitionCallback(t *testing.T) {
	s := NewStore(20)

	var transitions []StateTransition
	s.SetTransitionCallback(func(tr StateTransition) {
		transitions = append(transitions, tr)
		// Verify no deadlock when calling back into store from callback.
		_ = s.GetOperatingConditions()
	})

	s.SetOperatingConditions("test", OperatingConditions{
		ContextRetention:     0.5,
		MaxTokens:            1024,
		NoiseInjection:       0.2,
		InfoReorderIntensity: 0.3,
		Temperature:          0.9,
	}, "test update")

	if len(transitions) != 5 {
		t.Errorf("expected 5 transitions (all fields changed), got %d", len(transitions))
	}
}

func TestStore_TransitionCount(t *testing.T) {
	s := NewStore(20)

	if s.GetTransitionCount() != 0 {
		t.Errorf("expected 0 transitions initially, got %d", s.GetTransitionCount())
	}

	s.UpdateMetrics("test", func(m *Metrics) {
		m.Coherence = 0.5
	}, "change")

	if s.GetTransitionCount() != 1 {
		t.Errorf("expected 1 transition, got %d", s.GetTransitionCount())
	}
}

func TestStore_NaN_Rejected(t *testing.T) {
	s := NewStore(20)

	originalCoherence := s.GetMetrics().Coherence

	s.UpdateMetrics("test", func(m *Metrics) {
		m.Coherence = math.NaN()
	}, "NaN update")

	// NaN update should be rejected; metrics unchanged.
	if s.GetMetrics().Coherence != originalCoherence {
		t.Errorf("expected NaN update to be rejected, coherence changed to %f", s.GetMetrics().Coherence)
	}
}

func TestStore_Inf_Rejected(t *testing.T) {
	s := NewStore(20)

	originalCoherence := s.GetMetrics().Coherence

	s.UpdateMetrics("test", func(m *Metrics) {
		m.Coherence = math.Inf(1)
	}, "Inf update")

	if s.GetMetrics().Coherence != originalCoherence {
		t.Errorf("expected Inf update to be rejected, coherence changed to %f", s.GetMetrics().Coherence)
	}

	// Negative Inf too.
	s.UpdateMetrics("test", func(m *Metrics) {
		m.Coherence = math.Inf(-1)
	}, "-Inf update")

	if s.GetMetrics().Coherence != originalCoherence {
		t.Errorf("expected -Inf update to be rejected, coherence changed to %f", s.GetMetrics().Coherence)
	}
}

func TestStore_ConversationHistory_Retention(t *testing.T) {
	s := NewStore(20)

	// Add 10 turns (5 pairs).
	for i := 0; i < 5; i++ {
		s.AppendConversation(ConversationTurn{Role: "user", Content: "msg"})
		s.AppendConversation(ConversationTurn{Role: "assistant", Content: "resp"})
	}

	// Full retention should return all 10.
	h := s.GetConversationHistory(1.0)
	if len(h) != 10 {
		t.Errorf("expected 10 turns at full retention, got %d", len(h))
	}

	// 50% retention should return 4 (rounded down to even pair boundary).
	h = s.GetConversationHistory(0.5)
	if len(h)%2 != 0 {
		t.Errorf("retention should return even number of turns, got %d", len(h))
	}
	if h[0].Role != "user" {
		t.Errorf("retained history should start with user turn, got %q", h[0].Role)
	}

	// Zero retention should return nil.
	h = s.GetConversationHistory(0.0)
	if h != nil {
		t.Errorf("expected nil at zero retention, got %d turns", len(h))
	}

	// Negative retention should return nil.
	h = s.GetConversationHistory(-1.0)
	if h != nil {
		t.Errorf("expected nil at negative retention, got %d turns", len(h))
	}
}

func TestStore_ConversationHistory_MaxLimit(t *testing.T) {
	s := NewStore(6) // Max 6 turns

	// Add 10 turns (5 pairs).
	for i := 0; i < 5; i++ {
		s.AppendConversation(ConversationTurn{Role: "user", Content: "msg"})
		s.AppendConversation(ConversationTurn{Role: "assistant", Content: "resp"})
	}

	h := s.GetConversationHistory(1.0)
	if len(h) > 6 {
		t.Errorf("expected max 6 turns, got %d", len(h))
	}
	// Should start with a user turn (pair-aligned truncation).
	if len(h) > 0 && h[0].Role != "user" {
		t.Errorf("truncated history should start with user turn, got %q", h[0].Role)
	}
}

func TestStore_MinMaxHistory(t *testing.T) {
	// maxHistory < 2 should be clamped to 2.
	s := NewStore(0)

	s.AppendConversation(ConversationTurn{Role: "user", Content: "msg"})
	s.AppendConversation(ConversationTurn{Role: "assistant", Content: "resp"})
	s.AppendConversation(ConversationTurn{Role: "user", Content: "msg2"})
	s.AppendConversation(ConversationTurn{Role: "assistant", Content: "resp2"})

	h := s.GetConversationHistory(1.0)
	if len(h) > 2 {
		t.Errorf("expected max 2 turns (clamped), got %d", len(h))
	}
}

func TestStore_ConcurrentAccess(t *testing.T) {
	s := NewStore(100)

	// Register callback that calls back into store (deadlock regression test).
	s.SetTransitionCallback(func(tr StateTransition) {
		_ = s.GetMetrics()
		_ = s.GetOperatingConditions()
	})

	var wg sync.WaitGroup

	// Concurrent writers.
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				val := float64(j) / 100.0
				s.UpdateMetrics("test", func(m *Metrics) {
					m.Coherence = val
				}, "concurrent write")
			}
		}()
	}

	// Concurrent readers.
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_ = s.GetMetrics()
				_ = s.GetOperatingConditions()
				_ = s.GetConversationHistory(0.8)
				_ = s.Snapshot()
				_ = s.GetTransitionCount()
			}
		}()
	}

	wg.Wait()
}

func TestStore_Snapshot(t *testing.T) {
	s := NewStore(20)
	snap := s.Snapshot()

	if snap.Metrics.Coherence != 0.75 {
		t.Errorf("snapshot coherence mismatch: got %f", snap.Metrics.Coherence)
	}
	if snap.OperatingConditions.MaxTokens != 2048 {
		t.Errorf("snapshot max_tokens mismatch: got %d", snap.OperatingConditions.MaxTokens)
	}
	if snap.OperatingConditions.Temperature != 0.7 {
		t.Errorf("snapshot temperature mismatch: got %f", snap.OperatingConditions.Temperature)
	}
	if snap.ConversationLength != 0 {
		t.Errorf("expected empty conversation, got %d", snap.ConversationLength)
	}

	data, err := s.SnapshotJSON()
	if err != nil {
		t.Fatalf("SnapshotJSON failed: %v", err)
	}
	if len(data) == 0 {
		t.Error("empty JSON output")
	}
}
