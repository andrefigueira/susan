// Package orchestrator coordinates all subsystems of the ISC experiment.
//
// It manages the lifecycle of the Cognitive Core, Self-Monitor, and
// Homeostatic Regulator as concurrent goroutines, wires up the logging
// infrastructure, and runs the three-condition comparison protocol.
//
// Experimental design (three conditions):
//   - Control: Stateless single-shot API calls (no history, no feedback)
//   - History-only: Multi-turn with conversation history but NO feedback loop
//   - Architectured: Multi-turn with conversation history AND feedback loop
//
// The history-only condition is the critical control that isolates the effect
// of the feedback architecture from the effect of mere conversation context.
// Without it, observed differences between control and architectured could
// be explained entirely by the presence of conversation history.
package orchestrator

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/andrefigueira/susan/internal/claude"
	"github.com/andrefigueira/susan/internal/config"
	"github.com/andrefigueira/susan/internal/core"
	"github.com/andrefigueira/susan/internal/logging"
	"github.com/andrefigueira/susan/internal/monitor"
	"github.com/andrefigueira/susan/internal/regulator"
	"github.com/andrefigueira/susan/internal/scenarios"
	"github.com/andrefigueira/susan/internal/state"
)

// ExperimentResult captures the outcome of a single experiment run.
type ExperimentResult struct {
	RunID           string                    `json:"run_id"`
	Mode            string                    `json:"mode"`
	Scenario        string                    `json:"scenario"`
	Repetition      int                       `json:"repetition"`
	ConfigHash      string                    `json:"config_hash"`
	StartTime       time.Time                 `json:"start_time"`
	EndTime         time.Time                 `json:"end_time"`
	TaskOutputs     []core.TaskOutput         `json:"task_outputs"`
	FinalMetrics    state.Metrics             `json:"final_metrics"`
	FinalConditions state.OperatingConditions  `json:"final_conditions"`
	TransitionCount int                       `json:"transition_count"`
}

// Orchestrator manages the full experiment lifecycle.
type Orchestrator struct {
	cfg    *config.Config
	client *claude.Client
	logger *slog.Logger
}

// New creates an Orchestrator from configuration.
func New(cfg *config.Config) *Orchestrator {
	client := claude.NewClient(cfg.API.Key, cfg.API.BaseURL, cfg.API.Model)
	handler := logging.NewSlogHandler(cfg.Logging.Level)
	logger := slog.New(handler)

	return &Orchestrator{
		cfg:    cfg,
		client: client,
		logger: logger,
	}
}

// RunExperiment executes the full three-condition comparison protocol.
func (o *Orchestrator) RunExperiment(ctx context.Context) error {
	allScenarios := scenarios.DefaultScenarios()

	for rep := 0; rep < o.cfg.Experiment.Repetitions; rep++ {
		for _, scenario := range allScenarios {
			seed := o.cfg.Experiment.Seed + int64(rep*1000)

			// Run all three conditions for each scenario.
			for _, mode := range []string{"control", "history_only", "architectured"} {
				runID := fmt.Sprintf("%s_%s_rep%d_%d", mode, scenario.Name, rep, time.Now().Unix())
				o.logger.Info("starting run",
					"mode", mode,
					"scenario", scenario.Name,
					"repetition", rep,
					"run_id", runID,
				)

				result, err := o.runScenario(ctx, runID, scenario, mode, seed)
				if err != nil {
					o.logger.Error("run failed", "error", err, "mode", mode, "scenario", scenario.Name)
					return fmt.Errorf("%s run %s failed: %w", mode, scenario.Name, err)
				}
				o.logger.Info("run complete",
					"mode", mode,
					"scenario", scenario.Name,
					"tasks_completed", len(result.TaskOutputs),
				)
			}
		}
	}

	return nil
}

// RunSingleScenario runs a single scenario in the specified mode.
func (o *Orchestrator) RunSingleScenario(ctx context.Context, scenarioName string, mode string) (*ExperimentResult, error) {
	allScenarios := scenarios.DefaultScenarios()
	var target *scenarios.Scenario
	for i := range allScenarios {
		if allScenarios[i].Name == scenarioName {
			target = &allScenarios[i]
			break
		}
	}
	if target == nil {
		return nil, fmt.Errorf("scenario %q not found", scenarioName)
	}

	runID := fmt.Sprintf("%s_%s_%d", mode, scenarioName, time.Now().Unix())
	return o.runScenario(ctx, runID, *target, mode, o.cfg.Experiment.Seed)
}

// runScenario dispatches to the appropriate run mode.
func (o *Orchestrator) runScenario(ctx context.Context, runID string, scenario scenarios.Scenario, mode string, seed int64) (*ExperimentResult, error) {
	switch mode {
	case "control":
		return o.runControl(ctx, runID, scenario, seed)
	case "history_only":
		return o.runHistoryOnly(ctx, runID, scenario, seed)
	case "architectured":
		return o.runArchitectured(ctx, runID, scenario, seed)
	default:
		return nil, fmt.Errorf("unknown mode: %s (valid: control, history_only, architectured)", mode)
	}
}

// runControl executes a scenario in control mode: stateless, no feedback, no history.
func (o *Orchestrator) runControl(ctx context.Context, runID string, scenario scenarios.Scenario, seed int64) (*ExperimentResult, error) {
	expLog, err := logging.NewExperimentLogger(runID, o.cfg.Experiment.OutputDir, o.cfg.Logging, o.logger)
	if err != nil {
		return nil, fmt.Errorf("creating logger: %w", err)
	}
	defer expLog.Close()

	store := state.NewStore(o.cfg.CognitiveCore.MaxConversationHistory)
	cogCore := core.New(o.client, store, o.cfg.CognitiveCore.SystemPrompt, o.logger, seed)

	cogCore.SetOutputCallback(func(output core.TaskOutput) {
		expLog.Log(logging.EventCoreOutput, output)
	})

	result := &ExperimentResult{
		RunID:      runID,
		Mode:       "control",
		Scenario:   scenario.Name,
		ConfigHash: o.cfg.Hash(),
		StartTime:  time.Now(),
	}

	interTaskDelay := o.matchedInterTaskDelay()

	for i, task := range scenario.Tasks {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		default:
		}

		output, err := cogCore.ProcessControl(ctx, task)
		if err != nil {
			o.logger.Error("control task failed", "task_id", task.ID, "error", err)
			expLog.Log(logging.EventTaskFailure, logging.TaskFailure{
				TaskID: task.ID, Mode: "control", Error: err.Error(), Category: task.Category,
			})
			continue
		}
		result.TaskOutputs = append(result.TaskOutputs, *output)

		// Matched inter-task delay to prevent timing confounds with
		// the architectured condition (which waits for monitor + regulator).
		if i < len(scenario.Tasks)-1 {
			select {
			case <-ctx.Done():
				return result, ctx.Err()
			case <-time.After(interTaskDelay):
			}
		}
	}

	result.EndTime = time.Now()
	expLog.Log(logging.EventExperimentResult, result)
	return result, nil
}

// runHistoryOnly executes a scenario with conversation history but NO feedback loop.
// This is the critical control condition that isolates the feedback architecture's
// effect from the effect of mere multi-turn context.
func (o *Orchestrator) runHistoryOnly(ctx context.Context, runID string, scenario scenarios.Scenario, seed int64) (*ExperimentResult, error) {
	expLog, err := logging.NewExperimentLogger(runID, o.cfg.Experiment.OutputDir, o.cfg.Logging, o.logger)
	if err != nil {
		return nil, fmt.Errorf("creating logger: %w", err)
	}
	defer expLog.Close()

	store := state.NewStore(o.cfg.CognitiveCore.MaxConversationHistory)
	cogCore := core.New(o.client, store, o.cfg.CognitiveCore.SystemPrompt, o.logger, seed)

	cogCore.SetOutputCallback(func(output core.TaskOutput) {
		expLog.Log(logging.EventCoreOutput, output)
	})

	result := &ExperimentResult{
		RunID:      runID,
		Mode:       "history_only",
		Scenario:   scenario.Name,
		ConfigHash: o.cfg.Hash(),
		StartTime:  time.Now(),
	}

	interTaskDelay := o.matchedInterTaskDelay()

	for i, task := range scenario.Tasks {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		default:
		}

		output, err := cogCore.ProcessHistoryOnly(ctx, task)
		if err != nil {
			o.logger.Error("history-only task failed", "task_id", task.ID, "error", err)
			expLog.Log(logging.EventTaskFailure, logging.TaskFailure{
				TaskID: task.ID, Mode: "history_only", Error: err.Error(), Category: task.Category,
			})
			continue
		}
		result.TaskOutputs = append(result.TaskOutputs, *output)

		// Matched inter-task delay to prevent timing confounds with
		// the architectured condition.
		if i < len(scenario.Tasks)-1 {
			select {
			case <-ctx.Done():
				return result, ctx.Err()
			case <-time.After(interTaskDelay):
			}
		}
	}

	result.EndTime = time.Now()
	expLog.Log(logging.EventExperimentResult, result)
	return result, nil
}

// runArchitectured executes a scenario with the full feedback architecture.
func (o *Orchestrator) runArchitectured(ctx context.Context, runID string, scenario scenarios.Scenario, seed int64) (*ExperimentResult, error) {
	expLog, err := logging.NewExperimentLogger(runID, o.cfg.Experiment.OutputDir, o.cfg.Logging, o.logger)
	if err != nil {
		return nil, fmt.Errorf("creating logger: %w", err)
	}
	defer expLog.Close()

	store := state.NewStore(o.cfg.CognitiveCore.MaxConversationHistory)

	store.SetTransitionCallback(func(t state.StateTransition) {
		expLog.Log(logging.EventStateTransition, t)
	})

	cogCore := core.New(o.client, store, o.cfg.CognitiveCore.SystemPrompt, o.logger, seed)
	selfMon := monitor.New(o.client, store, o.cfg.SelfMonitor.SystemPrompt, o.cfg.SelfMonitor.MaxTokens, o.logger)
	reg := regulator.New(o.cfg.Homeostasis, o.cfg.Disruption, store, o.logger)

	cogCore.SetOutputCallback(func(output core.TaskOutput) {
		expLog.Log(logging.EventCoreOutput, output)
	})
	selfMon.SetAssessmentCallback(func(a monitor.TimestampedAssessment) {
		expLog.Log(logging.EventMonitorAssessment, a)
	})
	reg.SetActionCallback(func(a regulator.Action) {
		expLog.Log(logging.EventRegulatorAction, a)
	})

	subsysCtx, subsysCancel := context.WithCancel(ctx)

	// WaitGroup for proper goroutine shutdown.
	var wg sync.WaitGroup

	// Start Self-Monitor with panic recovery.
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				o.logger.Error("self-monitor panicked", "recover", r)
			}
		}()
		selfMon.Run(subsysCtx, time.Duration(o.cfg.TickRates.SelfMonitorMs)*time.Millisecond)
	}()

	// Start Homeostatic Regulator with panic recovery.
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				o.logger.Error("homeostatic regulator panicked", "recover", r)
			}
		}()
		reg.Run(subsysCtx, time.Duration(o.cfg.TickRates.HomeostaticRegulatorMs)*time.Millisecond)
	}()

	// Start state snapshot logger with panic recovery.
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				o.logger.Error("state snapshot logger panicked", "recover", r)
			}
		}()
		ticker := time.NewTicker(time.Duration(o.cfg.TickRates.StateLogMs) * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-subsysCtx.Done():
				return
			case <-ticker.C:
				expLog.Log(logging.EventStateSnapshot, store.Snapshot())
			}
		}
	}()

	result := &ExperimentResult{
		RunID:      runID,
		Mode:       "architectured",
		Scenario:   scenario.Name,
		ConfigHash: o.cfg.Hash(),
		StartTime:  time.Now(),
	}

	for _, task := range scenario.Tasks {
		select {
		case <-ctx.Done():
			subsysCancel()
			wg.Wait()
			return result, ctx.Err()
		default:
		}

		output, err := cogCore.Process(ctx, task, "architectured")
		if err != nil {
			o.logger.Error("architectured task failed", "task_id", task.ID, "error", err)
			expLog.Log(logging.EventTaskFailure, logging.TaskFailure{
				TaskID: task.ID, Mode: "architectured", Error: err.Error(), Category: task.Category,
			})
			continue
		}
		result.TaskOutputs = append(result.TaskOutputs, *output)

		// Feed the output to the Self-Monitor for evaluation.
		selfMon.SetLatestOutput(output.Response, task.ID, task.Prompt)

		// Wait for the monitor to complete its evaluation before proceeding.
		// This ensures the feedback loop has propagated: monitor evaluates ->
		// writes metrics -> regulator reads metrics -> adjusts conditions ->
		// next task uses adjusted conditions.
		//
		// We wait for the monitor's done signal OR a timeout. The timeout
		// accounts for slow API responses or skipped evaluations.
		monitorTimeout := time.Duration(o.cfg.TickRates.SelfMonitorMs)*time.Millisecond*3 + 30*time.Second
		select {
		case <-ctx.Done():
			subsysCancel()
			wg.Wait()
			return result, ctx.Err()
		case <-selfMon.DoneCh():
			// Monitor has completed evaluation. Wait one more regulator tick
			// to allow conditions to be updated.
			regulatorWait := time.Duration(o.cfg.TickRates.HomeostaticRegulatorMs) * time.Millisecond
			select {
			case <-ctx.Done():
				subsysCancel()
				wg.Wait()
				return result, ctx.Err()
			case <-time.After(regulatorWait + 200*time.Millisecond):
			}
		case <-time.After(monitorTimeout):
			o.logger.Warn("monitor evaluation timed out, proceeding", "task_id", task.ID)
		}
	}

	result.FinalMetrics = store.GetMetrics()
	result.FinalConditions = store.GetOperatingConditions()
	result.TransitionCount = store.GetTransitionCount()
	result.EndTime = time.Now()

	// Stop background subsystems and wait for them to finish.
	subsysCancel()
	wg.Wait()

	expLog.Log(logging.EventExperimentResult, result)
	return result, nil
}

// matchedInterTaskDelay returns the delay to insert between tasks in control
// and history_only modes. This matches the expected delay in architectured mode
// (monitor evaluation + regulator tick) to prevent timing confounds between
// experimental conditions.
func (o *Orchestrator) matchedInterTaskDelay() time.Duration {
	return time.Duration(o.cfg.TickRates.SelfMonitorMs)*time.Millisecond +
		time.Duration(o.cfg.TickRates.HomeostaticRegulatorMs)*time.Millisecond +
		200*time.Millisecond
}

// ListScenarios returns the names and descriptions of available scenarios.
func ListScenarios() []struct {
	Name        string
	Category    string
	Description string
	Hypothesis  string
	TaskCount   int
} {
	allScenarios := scenarios.DefaultScenarios()
	var list []struct {
		Name        string
		Category    string
		Description string
		Hypothesis  string
		TaskCount   int
	}
	for _, s := range allScenarios {
		list = append(list, struct {
			Name        string
			Category    string
			Description string
			Hypothesis  string
			TaskCount   int
		}{
			Name:        s.Name,
			Category:    s.Category,
			Description: s.Description,
			Hypothesis:  s.Hypothesis,
			TaskCount:   len(s.Tasks),
		})
	}
	return list
}
