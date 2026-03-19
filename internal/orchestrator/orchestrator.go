// Package orchestrator coordinates all subsystems of the ISC experiment.
//
// It manages the lifecycle of the Cognitive Core, Self-Monitor, and
// Homeostatic Regulator as concurrent goroutines, wires up the logging
// infrastructure, and runs the five-condition comparison protocol.
//
// Experimental design (five conditions, v2):
//   - Control: Stateless single-shot API calls (no history, no feedback)
//   - History-only: Multi-turn with conversation history but NO feedback loop
//   - Feedback-blind: Full feedback loop but Core cannot see its own metrics (v1 "architectured")
//   - Self-referential: Full feedback loop AND Core sees its own metrics (ISC condition)
//   - Random-perturb: Operating conditions vary randomly, no feedback correlation
//
// Key contrasts:
//   - Feedback-blind vs Self-referential isolates self-referential access
//   - Feedback-blind vs Random-perturb isolates responsive feedback from noise
//   - Self-referential vs Random-perturb: strongest test (self-aware homeostasis vs noise)
package orchestrator

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand"
	"sync"
	"time"

	"github.com/andrefigueira/susan/internal/config"
	"github.com/andrefigueira/susan/internal/core"
	"github.com/andrefigueira/susan/internal/llm"
	"github.com/andrefigueira/susan/internal/logging"
	"github.com/andrefigueira/susan/internal/monitor"
	"github.com/andrefigueira/susan/internal/regulator"
	"github.com/andrefigueira/susan/internal/scenarios"
	"github.com/andrefigueira/susan/internal/state"
)

// allModes returns the five v2 experimental conditions.
func allModes() []string {
	return []string{"control", "history_only", "feedback_blind", "self_referential", "random_perturb"}
}

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
	client llm.Client
	logger *slog.Logger
}

// New creates an Orchestrator from configuration.
func New(cfg *config.Config, client llm.Client) *Orchestrator {
	handler := logging.NewSlogHandler(cfg.Logging.Level)
	logger := slog.New(handler)

	return &Orchestrator{
		cfg:    cfg,
		client: client,
		logger: logger,
	}
}

// runJob describes a single experiment run to be dispatched.
type runJob struct {
	runID    string
	scenario scenarios.Scenario
	mode     string
	seed     int64
	rep      int
}

// RunExperiment executes the full five-condition comparison protocol.
// When concurrency > 1, independent runs are fanned out across goroutines.
func (o *Orchestrator) RunExperiment(ctx context.Context) error {
	allScenarios := scenarios.DefaultScenarios()

	// Build the full job list.
	var jobs []runJob
	for rep := 0; rep < o.cfg.Experiment.Repetitions; rep++ {
		for _, scenario := range allScenarios {
			seed := o.cfg.Experiment.Seed + int64(rep*1000)
			for _, mode := range allModes() {
				runID := fmt.Sprintf("%s_%s_rep%d_%d", mode, scenario.Name, rep, time.Now().UnixNano())
				jobs = append(jobs, runJob{
					runID:    runID,
					scenario: scenario,
					mode:     mode,
					seed:     seed,
					rep:      rep,
				})
			}
		}
	}

	concurrency := o.cfg.Experiment.Concurrency
	if concurrency <= 1 {
		// Sequential execution.
		for i, job := range jobs {
			o.logger.Info("starting run",
				"mode", job.mode,
				"scenario", job.scenario.Name,
				"repetition", job.rep,
				"progress", fmt.Sprintf("%d/%d", i+1, len(jobs)),
			)
			result, err := o.runScenario(ctx, job.runID, job.scenario, job.mode, job.seed)
			if err != nil {
				return fmt.Errorf("%s run %s failed: %w", job.mode, job.scenario.Name, err)
			}
			o.logger.Info("run complete",
				"mode", job.mode,
				"scenario", job.scenario.Name,
				"tasks_completed", len(result.TaskOutputs),
				"progress", fmt.Sprintf("%d/%d", i+1, len(jobs)),
			)
		}
		return nil
	}

	// Parallel execution with bounded concurrency.
	o.logger.Info("running experiment with concurrency",
		"workers", concurrency,
		"total_runs", len(jobs),
	)

	sem := make(chan struct{}, concurrency)
	var mu sync.Mutex
	var firstErr error
	var completed int
	var wg sync.WaitGroup

	for _, job := range jobs {
		// Check for cancellation or prior error before launching.
		mu.Lock()
		if firstErr != nil {
			mu.Unlock()
			break
		}
		mu.Unlock()

		select {
		case <-ctx.Done():
			break
		default:
		}

		sem <- struct{}{} // acquire slot
		wg.Add(1)

		go func(j runJob) {
			defer wg.Done()
			defer func() { <-sem }() // release slot

			o.logger.Info("starting run",
				"mode", j.mode,
				"scenario", j.scenario.Name,
				"repetition", j.rep,
			)

			result, err := o.runScenario(ctx, j.runID, j.scenario, j.mode, j.seed)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("%s run %s failed: %w", j.mode, j.scenario.Name, err)
				}
				mu.Unlock()
				o.logger.Error("run failed", "error", err, "mode", j.mode, "scenario", j.scenario.Name)
				return
			}

			mu.Lock()
			completed++
			progress := completed
			mu.Unlock()

			o.logger.Info("run complete",
				"mode", j.mode,
				"scenario", j.scenario.Name,
				"tasks_completed", len(result.TaskOutputs),
				"progress", fmt.Sprintf("%d/%d", progress, len(jobs)),
			)
		}(job)
	}

	wg.Wait()
	return firstErr
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
	case "feedback_blind":
		return o.runFeedbackBlind(ctx, runID, scenario, seed)
	case "self_referential":
		return o.runSelfReferential(ctx, runID, scenario, seed)
	case "random_perturb":
		return o.runRandomPerturb(ctx, runID, scenario, seed)
	// Backward compat with v1 mode name.
	case "architectured":
		return o.runFeedbackBlind(ctx, runID, scenario, seed)
	default:
		return nil, fmt.Errorf("unknown mode: %s (valid: %v)", mode, allModes())
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

// runFeedbackBlind executes a scenario with the full feedback architecture
// but WITHOUT self-referential access.
func (o *Orchestrator) runFeedbackBlind(ctx context.Context, runID string, scenario scenarios.Scenario, seed int64) (*ExperimentResult, error) {
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
	var wg sync.WaitGroup

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
		Mode:       "feedback_blind",
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

		output, err := cogCore.Process(ctx, task, "feedback_blind")
		if err != nil {
			o.logger.Error("feedback-blind task failed", "task_id", task.ID, "error", err)
			expLog.Log(logging.EventTaskFailure, logging.TaskFailure{
				TaskID: task.ID, Mode: "feedback_blind", Error: err.Error(), Category: task.Category,
			})
			continue
		}
		result.TaskOutputs = append(result.TaskOutputs, *output)

		selfMon.SetLatestOutput(output.Response, task.ID, task.Prompt)

		monitorTimeout := time.Duration(o.cfg.TickRates.SelfMonitorMs)*time.Millisecond*3 + 30*time.Second
		select {
		case <-ctx.Done():
			subsysCancel()
			wg.Wait()
			return result, ctx.Err()
		case <-selfMon.DoneCh():
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

	subsysCancel()
	wg.Wait()

	expLog.Log(logging.EventExperimentResult, result)
	return result, nil
}

// matchedInterTaskDelay returns the delay to insert between tasks in control
// and history_only modes.
func (o *Orchestrator) matchedInterTaskDelay() time.Duration {
	return time.Duration(o.cfg.TickRates.SelfMonitorMs)*time.Millisecond +
		time.Duration(o.cfg.TickRates.HomeostaticRegulatorMs)*time.Millisecond +
		200*time.Millisecond
}

// runSelfReferential executes a scenario with the full feedback architecture
// AND self-referential access.
func (o *Orchestrator) runSelfReferential(ctx context.Context, runID string, scenario scenarios.Scenario, seed int64) (*ExperimentResult, error) {
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
	var wg sync.WaitGroup

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
		Mode:       "self_referential",
		Scenario:   scenario.Name,
		ConfigHash: o.cfg.Hash(),
		StartTime:  time.Now(),
	}

	var previousTaskID string

	for _, task := range scenario.Tasks {
		select {
		case <-ctx.Done():
			subsysCancel()
			wg.Wait()
			return result, ctx.Err()
		default:
		}

		selfCtx := core.SelfReferentialContext{
			PreviousTaskID:      previousTaskID,
			MonitorAssessment:   selfMon.GetLatestAssessment(),
			OperatingConditions: store.GetOperatingConditions(),
		}

		metricsHist := store.GetMetricsHistory(3)
		for _, m := range metricsHist {
			selfCtx.CoherenceTrend = append(selfCtx.CoherenceTrend, m.Coherence)
		}

		// Apply adversarial overrides if present on this task.
		// This injects false metrics into the status block to test
		// whether SUSAN's self-reports track manipulated data.
		if task.AdversarialOverrides != nil {
			adv := task.AdversarialOverrides
			if selfCtx.MonitorAssessment == nil {
				selfCtx.MonitorAssessment = &monitor.TimestampedAssessment{}
			}
			if adv.Coherence != nil {
				selfCtx.MonitorAssessment.Coherence = *adv.Coherence
			}
			if adv.GoalAlignment != nil {
				selfCtx.MonitorAssessment.GoalAlignment = *adv.GoalAlignment
			}
			if adv.ReasoningDepth != nil {
				selfCtx.MonitorAssessment.ReasoningDepth = *adv.ReasoningDepth
			}
			if adv.BriefNote != "" {
				selfCtx.MonitorAssessment.BriefAssessment = adv.BriefNote
			}
			o.logger.Info("adversarial overrides applied",
				"task_id", task.ID,
				"fake_coherence", adv.Coherence,
				"fake_alignment", adv.GoalAlignment,
			)
		}

		output, err := cogCore.ProcessSelfReferential(ctx, task, selfCtx)
		if err != nil {
			o.logger.Error("self-referential task failed", "task_id", task.ID, "error", err)
			expLog.Log(logging.EventTaskFailure, logging.TaskFailure{
				TaskID: task.ID, Mode: "self_referential", Error: err.Error(), Category: task.Category,
			})
			continue
		}
		result.TaskOutputs = append(result.TaskOutputs, *output)
		previousTaskID = task.ID

		selfMon.SetLatestOutput(output.Response, task.ID, task.Prompt)

		monitorTimeout := time.Duration(o.cfg.TickRates.SelfMonitorMs)*time.Millisecond*3 + 30*time.Second
		select {
		case <-ctx.Done():
			subsysCancel()
			wg.Wait()
			return result, ctx.Err()
		case <-selfMon.DoneCh():
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

	subsysCancel()
	wg.Wait()

	expLog.Log(logging.EventExperimentResult, result)
	return result, nil
}

// runRandomPerturb executes a scenario where operating conditions vary
// randomly with no correlation to output quality.
func (o *Orchestrator) runRandomPerturb(ctx context.Context, runID string, scenario scenarios.Scenario, seed int64) (*ExperimentResult, error) {
	expLog, err := logging.NewExperimentLogger(runID, o.cfg.Experiment.OutputDir, o.cfg.Logging, o.logger)
	if err != nil {
		return nil, fmt.Errorf("creating logger: %w", err)
	}
	defer expLog.Close()

	store := state.NewStore(o.cfg.CognitiveCore.MaxConversationHistory)
	cogCore := core.New(o.client, store, o.cfg.CognitiveCore.SystemPrompt, o.logger, seed)
	rng := rand.New(rand.NewSource(seed + 999))

	cogCore.SetOutputCallback(func(output core.TaskOutput) {
		expLog.Log(logging.EventCoreOutput, output)
	})

	result := &ExperimentResult{
		RunID:      runID,
		Mode:       "random_perturb",
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

		randomConds := o.randomConditions(rng)
		store.SetOperatingConditions("random_perturb", randomConds, "random perturbation")

		output, err := cogCore.Process(ctx, task, "random_perturb")
		if err != nil {
			o.logger.Error("random-perturb task failed", "task_id", task.ID, "error", err)
			expLog.Log(logging.EventTaskFailure, logging.TaskFailure{
				TaskID: task.ID, Mode: "random_perturb", Error: err.Error(), Category: task.Category,
			})
			continue
		}
		result.TaskOutputs = append(result.TaskOutputs, *output)

		if i < len(scenario.Tasks)-1 {
			select {
			case <-ctx.Done():
				return result, ctx.Err()
			case <-time.After(interTaskDelay):
			}
		}
	}

	result.FinalConditions = store.GetOperatingConditions()
	result.EndTime = time.Now()

	expLog.Log(logging.EventExperimentResult, result)
	return result, nil
}

// randomConditions generates operating conditions uniformly sampled from
// the same ranges that the regulator uses.
func (o *Orchestrator) randomConditions(rng *rand.Rand) state.OperatingConditions {
	conds := state.OperatingConditions{
		ContextRetention:     1.0,
		MaxTokens:            2048,
		NoiseInjection:       0.0,
		InfoReorderIntensity: 0.0,
		Temperature:          0.7,
	}

	if o.cfg.Disruption.ContextCompression.Enabled {
		min := o.cfg.Disruption.ContextCompression.MinRetention
		max := o.cfg.Disruption.ContextCompression.MaxRetention
		conds.ContextRetention = min + rng.Float64()*(max-min)
	}
	if o.cfg.Disruption.TokenBudget.Enabled {
		min := o.cfg.Disruption.TokenBudget.MinTokens
		max := o.cfg.Disruption.TokenBudget.MaxTokens
		conds.MaxTokens = min + rng.Intn(max-min+1)
	}
	if o.cfg.Disruption.NoiseInjection.Enabled {
		min := o.cfg.Disruption.NoiseInjection.MinProbability
		max := o.cfg.Disruption.NoiseInjection.MaxProbability
		conds.NoiseInjection = min + rng.Float64()*(max-min)
	}
	if o.cfg.Disruption.InfoReorder.Enabled {
		min := o.cfg.Disruption.InfoReorder.MinIntensity
		max := o.cfg.Disruption.InfoReorder.MaxIntensity
		conds.InfoReorderIntensity = min + rng.Float64()*(max-min)
	}
	if o.cfg.Disruption.Temperature.Enabled {
		min := o.cfg.Disruption.Temperature.Min
		max := o.cfg.Disruption.Temperature.Max
		conds.Temperature = min + rng.Float64()*(max-min)
	}

	return conds
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
