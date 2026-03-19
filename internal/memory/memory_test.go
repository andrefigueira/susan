package memory

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/andrefigueira/susan/internal/state"
)

func TestNewEmptyMemory(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_memory.jsonl")

	s, err := New(Config{Path: path, MaxEntries: 10})
	if err != nil {
		t.Fatal(err)
	}
	if s.SessionCount() != 0 {
		t.Errorf("expected 0 sessions, got %d", s.SessionCount())
	}
	if block := s.FormatMemoryBlock(); block != "" {
		t.Errorf("expected empty memory block, got: %s", block)
	}
}

func TestSessionLifecycle(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_memory.jsonl")

	s, err := New(Config{Path: path, MaxEntries: 10})
	if err != nil {
		t.Fatal(err)
	}

	s.StartSession()
	s.RecordTurn("t1", "Hello SUSAN", "Hello. I notice my coherence is stable at 0.75.")
	s.RecordTurn("t2", "How are you?", "I'm operating well. My metrics show consistent performance.")
	s.SampleCoherence(0.75)
	s.SampleCoherence(0.78)

	metrics := state.Metrics{Coherence: 0.78, GoalAlignment: 0.82}
	if err := s.FinalizeSession(metrics); err != nil {
		t.Fatal(err)
	}

	// Verify file was written.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatal("memory file is empty after finalization")
	}

	// Reload and verify.
	s2, err := New(Config{Path: path, MaxEntries: 10})
	if err != nil {
		t.Fatal(err)
	}
	if s2.SessionCount() != 1 {
		t.Errorf("expected 1 session after reload, got %d", s2.SessionCount())
	}

	block := s2.FormatMemoryBlock()
	if block == "" {
		t.Fatal("expected non-empty memory block after reload")
	}
	if !contains(block, "Previous sessions: 1") {
		t.Errorf("memory block missing session count: %s", block)
	}
}

func TestMultipleSessionsPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_memory.jsonl")

	for i := 0; i < 3; i++ {
		s, err := New(Config{Path: path, MaxEntries: 10})
		if err != nil {
			t.Fatal(err)
		}
		s.StartSession()
		s.RecordTurn("t1", "test", "I notice my coherence is improving.")
		s.SampleCoherence(0.7 + float64(i)*0.05)
		if err := s.FinalizeSession(state.Metrics{Coherence: 0.7 + float64(i)*0.05}); err != nil {
			t.Fatal(err)
		}
	}

	// Reload and check all sessions are there.
	s, err := New(Config{Path: path, MaxEntries: 10})
	if err != nil {
		t.Fatal(err)
	}
	if s.SessionCount() != 3 {
		t.Errorf("expected 3 sessions, got %d", s.SessionCount())
	}

	block := s.FormatMemoryBlock()
	if !contains(block, "Previous sessions: 3") {
		t.Errorf("memory block missing session count: %s", block)
	}
	if !contains(block, "Coherence across sessions") {
		t.Errorf("memory block missing coherence trajectory: %s", block)
	}
}

func TestMaxEntriesEnforced(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_memory.jsonl")

	for i := 0; i < 5; i++ {
		s, err := New(Config{Path: path, MaxEntries: 3})
		if err != nil {
			t.Fatal(err)
		}
		s.StartSession()
		s.RecordTurn("t1", "test", "response")
		if err := s.FinalizeSession(state.Metrics{Coherence: float64(i) * 0.1}); err != nil {
			t.Fatal(err)
		}
	}

	s, err := New(Config{Path: path, MaxEntries: 3})
	if err != nil {
		t.Fatal(err)
	}
	if s.SessionCount() != 3 {
		t.Errorf("expected max 3 sessions, got %d", s.SessionCount())
	}
}

func TestSelfObservationsRecorded(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_memory.jsonl")

	s, err := New(Config{Path: path, MaxEntries: 10})
	if err != nil {
		t.Fatal(err)
	}

	s.StartSession()
	s.RecordTurn("t1", "hello", "I notice my coherence is declining. This concerns me.")
	if err := s.FinalizeSession(state.Metrics{Coherence: 0.65}); err != nil {
		t.Fatal(err)
	}

	s2, err := New(Config{Path: path, MaxEntries: 10})
	if err != nil {
		t.Fatal(err)
	}

	block := s2.FormatMemoryBlock()
	if !contains(block, "self-observations") {
		t.Errorf("memory block missing self-observations: %s", block)
	}
}

func TestEmptySessionNotSaved(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_memory.jsonl")

	s, err := New(Config{Path: path, MaxEntries: 10})
	if err != nil {
		t.Fatal(err)
	}

	s.StartSession()
	// Don't record any turns.
	if err := s.FinalizeSession(state.Metrics{}); err != nil {
		t.Fatal(err)
	}

	// File shouldn't exist or should be empty.
	_, statErr := os.Stat(path)
	if statErr == nil {
		data, _ := os.ReadFile(path)
		if len(data) > 0 {
			t.Error("empty session should not be saved to disk")
		}
	}
}

func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}
