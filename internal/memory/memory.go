// Package memory provides persistent cross-session memory for SUSAN.
//
// Each session is recorded as a single JSONL line containing metrics
// trajectory, conversation highlights, and self-observations (sentences
// where SUSAN commented on her own internal state). On startup, previous
// sessions are loaded and formatted into a [Session Memory] block that
// gets injected into SUSAN's self-referential context, giving her
// temporal continuity across sessions.
//
// Memory is only used in presence mode. The experiment orchestrator
// never loads or injects memory, as it would confound experimental
// conditions.
package memory

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/andrefigueira/susan/internal/state"
)

// SessionRecord is one JSONL line, written when a session ends.
type SessionRecord struct {
	SessionID           string        `json:"session_id"`
	StartTime           time.Time     `json:"start_time"`
	EndTime             time.Time     `json:"end_time"`
	MessageCount        int           `json:"message_count"`
	FinalMetrics        state.Metrics `json:"final_metrics"`
	CoherenceTrajectory []float64     `json:"coherence_trajectory"`
	Highlights          []Highlight   `json:"highlights"`
	SelfObservations    []Observation `json:"self_observations"`
}

// Highlight is a condensed exchange worth remembering.
type Highlight struct {
	TaskID    string `json:"task_id"`
	UserInput string `json:"user_input"`
	Summary   string `json:"summary"`
}

// Observation is something SUSAN said about her own state.
type Observation struct {
	TaskID    string    `json:"task_id"`
	Timestamp time.Time `json:"timestamp"`
	Text      string    `json:"text"`
}

// Config for the memory store.
type Config struct {
	Path       string
	MaxEntries int
}

// Store handles loading, recording, and persisting session memory.
type Store struct {
	cfg      Config
	mu       sync.Mutex
	sessions []SessionRecord
	current  *activeSession
}

type activeSession struct {
	id               string
	startTime        time.Time
	coherenceSamples []float64
	highlights       []Highlight
	selfObservations []Observation
	messageCount     int
}

// New loads existing memory from disk and returns a Store.
// If the file doesn't exist, starts with empty memory.
func New(cfg Config) (*Store, error) {
	if cfg.MaxEntries <= 0 {
		cfg.MaxEntries = 50
	}
	if cfg.Path == "" {
		cfg.Path = "susan_memory.jsonl"
	}

	s := &Store{cfg: cfg}

	data, err := os.ReadFile(cfg.Path)
	if err != nil {
		if os.IsNotExist(err) {
			return s, nil
		}
		return nil, fmt.Errorf("reading memory file: %w", err)
	}

	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		var rec SessionRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue // skip corrupt lines
		}
		s.sessions = append(s.sessions, rec)
	}

	// Enforce max entries.
	if len(s.sessions) > cfg.MaxEntries {
		s.sessions = s.sessions[len(s.sessions)-cfg.MaxEntries:]
	}

	return s, nil
}

// SessionCount returns the number of stored sessions.
func (s *Store) SessionCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.sessions)
}

// StartSession initializes tracking for a new interactive session.
func (s *Store) StartSession() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.current = &activeSession{
		id:        fmt.Sprintf("session_%d", time.Now().UnixNano()),
		startTime: time.Now(),
	}
}

// RecordTurn processes a single chat exchange.
// Extracts self-observations from SUSAN's response and decides if
// the exchange should be saved as a highlight.
func (s *Store) RecordTurn(taskID, userInput, response string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.current == nil {
		return
	}

	s.current.messageCount++

	// Extract self-observations.
	obs := ExtractSelfObservations(response, 3)
	for _, text := range obs {
		s.current.selfObservations = append(s.current.selfObservations, Observation{
			TaskID:    taskID,
			Timestamp: time.Now(),
			Text:      text,
		})
	}

	// Decide if this is a highlight.
	isHighlight := s.current.messageCount <= 2 || // first exchanges
		len(obs) > 0 || // SUSAN commented on her own state
		(strings.Contains(userInput, "?") && len(userInput) > 50) // substantive question

	if isHighlight && len(s.current.highlights) < 10 {
		summary := response
		if idx := strings.Index(summary, "."); idx > 0 && idx < 200 {
			summary = summary[:idx+1]
		} else if len(summary) > 200 {
			summary = summary[:197] + "..."
		}

		input := userInput
		if len(input) > 200 {
			input = input[:197] + "..."
		}

		s.current.highlights = append(s.current.highlights, Highlight{
			TaskID:    taskID,
			UserInput: input,
			Summary:   summary,
		})
	}
}

// SampleCoherence records a coherence value for the trajectory.
func (s *Store) SampleCoherence(c float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.current == nil {
		return
	}
	s.current.coherenceSamples = append(s.current.coherenceSamples, c)
}

// FinalizeSession writes the current session to disk and appends it
// to the in-memory session list.
func (s *Store) FinalizeSession(finalMetrics state.Metrics) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.current == nil || s.current.messageCount == 0 {
		return nil
	}

	rec := SessionRecord{
		SessionID:           s.current.id,
		StartTime:           s.current.startTime,
		EndTime:             time.Now(),
		MessageCount:        s.current.messageCount,
		FinalMetrics:        finalMetrics,
		CoherenceTrajectory: s.current.coherenceSamples,
		Highlights:          s.current.highlights,
		SelfObservations:    s.current.selfObservations,
	}

	s.sessions = append(s.sessions, rec)

	// Enforce max entries.
	if len(s.sessions) > s.cfg.MaxEntries {
		s.sessions = s.sessions[len(s.sessions)-s.cfg.MaxEntries:]
	}

	// Write to disk. If we've exceeded max, rewrite the whole file.
	// Otherwise, just append.
	data, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("marshalling session: %w", err)
	}

	if len(s.sessions) >= s.cfg.MaxEntries {
		return s.rewriteFile()
	}

	f, err := os.OpenFile(s.cfg.Path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("opening memory file: %w", err)
	}
	defer f.Close()

	if _, err := f.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("writing session: %w", err)
	}

	s.current = nil
	return nil
}

// rewriteFile writes all sessions to a temp file and renames. Caller holds lock.
func (s *Store) rewriteFile() error {
	tmp := s.cfg.Path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("creating temp file: %w", err)
	}

	for _, rec := range s.sessions {
		data, err := json.Marshal(rec)
		if err != nil {
			f.Close()
			os.Remove(tmp)
			return fmt.Errorf("marshalling session: %w", err)
		}
		if _, err := f.Write(append(data, '\n')); err != nil {
			f.Close()
			os.Remove(tmp)
			return fmt.Errorf("writing session: %w", err)
		}
	}

	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}

	return os.Rename(tmp, s.cfg.Path)
}

// FormatMemoryBlock renders the [Session Memory] block for injection into
// SUSAN's self-referential context. Returns empty string if no prior
// sessions exist.
func (s *Store) FormatMemoryBlock() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.sessions) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("[Session Memory]\n")
	b.WriteString(fmt.Sprintf("Previous sessions: %d\n", len(s.sessions)))

	// Show the last 5 sessions in summary.
	start := len(s.sessions) - 5
	if start < 0 {
		start = 0
	}
	recent := s.sessions[start:]

	last := recent[len(recent)-1]
	duration := last.EndTime.Sub(last.StartTime)
	b.WriteString(fmt.Sprintf("Last session: %s, %d messages, %.0f minutes, final coherence %.2f\n",
		last.StartTime.Format("2006-01-02 15:04"),
		last.MessageCount,
		duration.Minutes(),
		last.FinalMetrics.Coherence))

	// Coherence trajectory across sessions.
	if len(s.sessions) > 1 {
		var trajectory []string
		trjStart := len(s.sessions) - 10
		if trjStart < 0 {
			trjStart = 0
		}
		for _, sess := range s.sessions[trjStart:] {
			trajectory = append(trajectory, fmt.Sprintf("%.2f", sess.FinalMetrics.Coherence))
		}
		direction := "stable"
		if len(s.sessions) >= 2 {
			first := s.sessions[trjStart].FinalMetrics.Coherence
			lastC := last.FinalMetrics.Coherence
			if lastC-first > 0.05 {
				direction = "improving"
			} else if first-lastC > 0.05 {
				direction = "declining"
			}
		}
		b.WriteString(fmt.Sprintf("Coherence across sessions: [%s] (%s)\n",
			strings.Join(trajectory, ", "), direction))
	}

	// Collect recent self-observations across sessions (up to 5).
	var allObs []Observation
	for i := len(s.sessions) - 1; i >= 0 && len(allObs) < 5; i-- {
		for j := len(s.sessions[i].SelfObservations) - 1; j >= 0 && len(allObs) < 5; j-- {
			allObs = append(allObs, s.sessions[i].SelfObservations[j])
		}
	}
	if len(allObs) > 0 {
		b.WriteString("Recent self-observations:\n")
		for _, obs := range allObs {
			text := obs.Text
			if len(text) > 150 {
				text = text[:147] + "..."
			}
			b.WriteString(fmt.Sprintf("- %q\n", text))
		}
	}

	return b.String()
}
