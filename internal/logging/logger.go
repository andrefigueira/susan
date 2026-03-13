// Package logging provides structured JSON logging for all experiment data.
//
// Each event type writes to a separate JSONL file for clean downstream analysis.
// All log entries include a monotonic sequence number and wall-clock timestamp
// to support causal ordering of events across concurrent subsystems.
package logging

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/andrefigueira/susan/internal/config"
)

// EventType identifies the category of logged event.
type EventType string

const (
	EventStateTransition   EventType = "state_transition"
	EventCoreOutput        EventType = "core_output"
	EventMonitorAssessment EventType = "monitor_assessment"
	EventRegulatorAction   EventType = "regulator_action"
	EventExperimentResult  EventType = "experiment_result"
	EventStateSnapshot     EventType = "state_snapshot"
	EventTaskFailure       EventType = "task_failure"
)

// Entry is the wrapper for all logged events.
type Entry struct {
	Sequence  uint64      `json:"seq"`
	Timestamp time.Time   `json:"timestamp"`
	Type      EventType   `json:"type"`
	RunID     string      `json:"run_id"`
	Data      interface{} `json:"data"`
}

// TaskFailure records a failed task attempt.
type TaskFailure struct {
	TaskID   string `json:"task_id"`
	Mode     string `json:"mode"`
	Error    string `json:"error"`
	Category string `json:"category"`
}

// ExperimentLogger manages structured logging for a single experiment run.
type ExperimentLogger struct {
	runID     string
	seq       atomic.Uint64
	outputDir string
	logger    *slog.Logger

	mu    sync.Mutex
	files map[EventType]*os.File
}

// NewExperimentLogger creates a logger for a specific experiment run.
func NewExperimentLogger(runID string, outputDir string, logCfg config.LoggingConfig, logger *slog.Logger) (*ExperimentLogger, error) {
	runDir := filepath.Join(outputDir, runID)
	if err := os.MkdirAll(runDir, 0755); err != nil {
		return nil, fmt.Errorf("creating log directory: %w", err)
	}

	el := &ExperimentLogger{
		runID:     runID,
		outputDir: runDir,
		logger:    logger,
		files:     make(map[EventType]*os.File),
	}

	fileMap := map[EventType]string{
		EventStateTransition:   logCfg.Files.StateTransitions,
		EventCoreOutput:        logCfg.Files.CoreOutputs,
		EventMonitorAssessment: logCfg.Files.MonitorAssessments,
		EventRegulatorAction:   logCfg.Files.RegulatorActions,
		EventExperimentResult:  logCfg.Files.ExperimentResults,
		EventStateSnapshot:     "state_snapshots.jsonl",
		EventTaskFailure:       "task_failures.jsonl",
	}

	for eventType, filename := range fileMap {
		f, err := os.Create(filepath.Join(runDir, filename))
		if err != nil {
			el.Close()
			return nil, fmt.Errorf("creating %s: %w", filename, err)
		}
		el.files[eventType] = f
	}

	return el, nil
}

// Log writes an event to the appropriate log file.
// Errors are logged to slog rather than silently discarded.
func (el *ExperimentLogger) Log(eventType EventType, data interface{}) {
	entry := Entry{
		Sequence:  el.seq.Add(1),
		Timestamp: time.Now(),
		Type:      eventType,
		RunID:     el.runID,
		Data:      data,
	}

	b, err := json.Marshal(entry)
	if err != nil {
		el.logger.Error("failed to marshal log entry", "type", eventType, "error", err)
		return
	}

	el.mu.Lock()
	defer el.mu.Unlock()

	f, ok := el.files[eventType]
	if !ok {
		el.logger.Error("unknown event type for logging", "type", eventType)
		return
	}

	if _, err := f.Write(append(b, '\n')); err != nil {
		el.logger.Error("failed to write log entry", "type", eventType, "error", err)
	}
}

// Close flushes and closes all log files.
func (el *ExperimentLogger) Close() error {
	el.mu.Lock()
	defer el.mu.Unlock()

	var firstErr error
	for _, f := range el.files {
		if err := f.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// OutputDir returns the directory where logs are written.
func (el *ExperimentLogger) OutputDir() string {
	return el.outputDir
}

// NewSlogHandler creates an slog.Handler that writes to stderr with
// the configured log level.
func NewSlogHandler(level string) slog.Handler {
	var logLevel slog.Level
	switch level {
	case "debug":
		logLevel = slog.LevelDebug
	case "info":
		logLevel = slog.LevelInfo
	case "warn":
		logLevel = slog.LevelWarn
	case "error":
		logLevel = slog.LevelError
	default:
		logLevel = slog.LevelInfo
	}

	return slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
		Level: logLevel,
	})
}
