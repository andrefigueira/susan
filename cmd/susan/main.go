// Susan -- ISC Consciousness Architecture Experiment
//
// Named after Susan Calvin, Asimov's robopsychologist, this system tests
// predictions from Informational Substrate Convergence theory by comparing
// feedback-architectured LLM systems against baseline API calls.
//
// Five experimental conditions:
//
//	susan run --mode control           Stateless single-shot (no history, no feedback)
//	susan run --mode history_only      Multi-turn with history (no feedback)
//	susan run --mode feedback_blind    Full feedback loop, Core cannot see metrics
//	susan run --mode self_referential  Full feedback loop AND Core sees own metrics
//	susan run --mode random_perturb    Random parameter variation, no feedback
//
// Usage:
//
//	susan run                     Run full experiment (all conditions, all scenarios)
//	susan run --scenario NAME     Run a single scenario
//	susan run --mode MODE         Run only one condition
//	susan evaluate                Run blind post-hoc evaluation
//	susan export-for-rating       Export blinded responses as CSV for human raters
//	susan import-ratings          Import completed human rating CSVs
//	susan scenarios               List available test scenarios
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/andrefigueira/susan/internal/config"
	"github.com/andrefigueira/susan/internal/evaluate"
	"github.com/andrefigueira/susan/internal/llm"
	"github.com/andrefigueira/susan/internal/logging"
	"github.com/andrefigueira/susan/internal/orchestrator"
	"github.com/andrefigueira/susan/internal/presence"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "run":
		runCmd(os.Args[2:])
	case "evaluate":
		evaluateCmd(os.Args[2:])
	case "export-for-rating":
		exportForRatingCmd(os.Args[2:])
	case "import-ratings":
		importRatingsCmd(os.Args[2:])
	case "presence":
		presenceCmd(os.Args[2:])
	case "scenarios":
		listScenarios()
	case "help", "--help", "-h":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

// newClient creates an LLM client from the API config.
func newClient(api config.APIConfig) llm.Client {
	switch api.Provider {
	case "openai":
		return llm.NewOpenAIClient(api.Key, api.BaseURL, api.Model)
	default:
		return llm.NewAnthropicClient(api.Key, api.BaseURL, api.Model)
	}
}

// newEvalClient creates an LLM client for the blind evaluator.
func newEvalClient(cfg *config.Config) llm.Client {
	evalModel := cfg.API.EvaluatorModel
	if evalModel == "" {
		evalModel = cfg.API.Model
	}
	switch cfg.API.Provider {
	case "openai":
		return llm.NewOpenAIClient(cfg.API.Key, cfg.API.BaseURL, evalModel)
	default:
		return llm.NewAnthropicClient(cfg.API.Key, cfg.API.BaseURL, evalModel)
	}
}

// newCrossEvalClient creates an optional cross-model evaluator client.
func newCrossEvalClient(cfg config.CrossEvaluationConfig) llm.Client {
	switch cfg.Provider {
	case "openai":
		return llm.NewOpenAIClient(cfg.Key, cfg.BaseURL, cfg.Model)
	default:
		return llm.NewAnthropicClient(cfg.Key, cfg.BaseURL, cfg.Model)
	}
}

func runCmd(args []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	configPath := fs.String("config", "config.yaml", "path to configuration file")
	scenario := fs.String("scenario", "", "run a single scenario by name")
	mode := fs.String("mode", "", "run mode: control, history_only, feedback_blind, self_referential, random_perturb, or all (default)")
	fs.Parse(args)

	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading config: %v\n", err)
		os.Exit(1)
	}

	handler := logging.NewSlogHandler(cfg.Logging.Level)
	logger := slog.New(handler)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		logger.Info("received signal, shutting down", "signal", sig)
		cancel()
	}()

	client := newClient(cfg.API)
	orch := orchestrator.New(cfg, client)

	if *scenario != "" {
		modes := []string{"control", "history_only", "feedback_blind", "self_referential", "random_perturb"}
		if *mode != "" {
			modes = []string{*mode}
		}

		for _, m := range modes {
			logger.Info("running scenario", "scenario", *scenario, "mode", m)
			result, err := orch.RunSingleScenario(ctx, *scenario, m)
			if err != nil {
				logger.Error("scenario failed", "error", err)
				os.Exit(1)
			}
			fmt.Printf("\n=== %s [%s] ===\n", result.Scenario, result.Mode)
			fmt.Printf("Tasks completed: %d\n", len(result.TaskOutputs))
			fmt.Printf("Duration: %s\n", result.EndTime.Sub(result.StartTime))
			if result.Mode == "feedback_blind" || result.Mode == "self_referential" {
				fmt.Printf("Final coherence: %.3f\n", result.FinalMetrics.Coherence)
				fmt.Printf("Final disruption: %.3f\n", result.FinalMetrics.DisruptionLevel)
				fmt.Printf("State transitions: %d\n", result.TransitionCount)
			}
			fmt.Printf("Logs: %s/\n", cfg.Experiment.OutputDir)
		}
	} else {
		logger.Info("starting full experiment",
			"scenarios", len(orchestrator.ListScenarios()),
			"repetitions", cfg.Experiment.Repetitions,
			"conditions", 5,
		)

		if err := orch.RunExperiment(ctx); err != nil {
			logger.Error("experiment failed", "error", err)
			os.Exit(1)
		}

		logger.Info("experiment complete")
		fmt.Printf("\nExperiment complete. Logs written to: %s/\n", cfg.Experiment.OutputDir)
	}
}

func evaluateCmd(args []string) {
	fs := flag.NewFlagSet("evaluate", flag.ExitOnError)
	configPath := fs.String("config", "config.yaml", "path to configuration file")
	experimentDir := fs.String("dir", "", "path to experiment output directory (required)")
	outputDir := fs.String("output", "", "path to write evaluation results (default: <dir>/blind_eval)")
	fs.Parse(args)

	if *experimentDir == "" {
		fmt.Fprintf(os.Stderr, "error: --dir is required (path to experiment output directory)\n")
		fmt.Fprintf(os.Stderr, "usage: susan evaluate --dir ./experiments\n")
		os.Exit(1)
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading config: %v\n", err)
		os.Exit(1)
	}

	handler := logging.NewSlogHandler(cfg.Logging.Level)
	logger := slog.New(handler)

	evalOutput := *outputDir
	if evalOutput == "" {
		evalOutput = *experimentDir + "/blind_eval"
	}
	if err := os.MkdirAll(evalOutput, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "error creating output dir: %v\n", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		logger.Info("received signal, shutting down", "signal", sig)
		cancel()
	}()

	evalClient := newEvalClient(cfg)
	ev := evaluate.NewEvaluator(evalClient, logger, evalOutput)

	if cfg.CrossEvaluation.Enabled {
		crossClient := newCrossEvalClient(cfg.CrossEvaluation)
		ev.SetCrossModelClient(crossClient)
		fmt.Printf("  Cross-model evaluator: %s (%s)\n", cfg.CrossEvaluation.Model, cfg.CrossEvaluation.Provider)
	}

	fmt.Println("Starting blind evaluation pipeline...")
	fmt.Printf("  Experiment dir: %s\n", *experimentDir)
	fmt.Printf("  Output dir:     %s\n", evalOutput)
	fmt.Println()

	if err := ev.Run(ctx, *experimentDir); err != nil {
		logger.Error("evaluation failed", "error", err)
		os.Exit(1)
	}

	fmt.Printf("\nBlind evaluation complete.\n")
	fmt.Printf("  Blind evaluations: %s/blind_evaluations.jsonl\n", evalOutput)
	fmt.Printf("  Blind mapping:     %s/blind_mapping.jsonl\n", evalOutput)
	fmt.Printf("  De-blinded:        %s/deblinded_evaluations.jsonl\n", evalOutput)
	fmt.Printf("\nRun analysis: python3 analysis/analyse.py %s\n", *experimentDir)
}

func exportForRatingCmd(args []string) {
	fs := flag.NewFlagSet("export-for-rating", flag.ExitOnError)
	configPath := fs.String("config", "config.yaml", "path to configuration file")
	experimentDir := fs.String("dir", "", "path to experiment output directory (required)")
	outputPath := fs.String("output", "human_ratings.csv", "output CSV path")
	fs.Parse(args)

	if *experimentDir == "" {
		fmt.Fprintf(os.Stderr, "error: --dir is required\n")
		os.Exit(1)
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading config: %v\n", err)
		os.Exit(1)
	}

	handler := logging.NewSlogHandler(cfg.Logging.Level)
	logger := slog.New(handler)

	ev := evaluate.NewEvaluator(newEvalClient(cfg), logger, "")

	if err := ev.ExportForHumanRating(*experimentDir, *outputPath); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Exported blinded responses to: %s\n", *outputPath)
	fmt.Println("Distribute this CSV to human raters. They fill in the blank columns.")
	fmt.Printf("Import completed ratings with: susan import-ratings --csv %s\n", *outputPath)
}

func importRatingsCmd(args []string) {
	fs := flag.NewFlagSet("import-ratings", flag.ExitOnError)
	configPath := fs.String("config", "config.yaml", "path to configuration file")
	csvPath := fs.String("csv", "", "path to completed human ratings CSV (required)")
	outputPath := fs.String("output", "human_evaluations.jsonl", "output JSONL path")
	fs.Parse(args)

	if *csvPath == "" {
		fmt.Fprintf(os.Stderr, "error: --csv is required\n")
		os.Exit(1)
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading config: %v\n", err)
		os.Exit(1)
	}

	handler := logging.NewSlogHandler(cfg.Logging.Level)
	logger := slog.New(handler)

	ev := evaluate.NewEvaluator(newEvalClient(cfg), logger, "")

	if err := ev.ImportHumanRatings(*csvPath, *outputPath); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Imported human ratings to: %s\n", *outputPath)
}

func presenceCmd(args []string) {
	fs := flag.NewFlagSet("presence", flag.ExitOnError)
	configPath := fs.String("config", "config.yaml", "path to configuration file")
	port := fs.Int("port", 0, "override port (default: config or 3000)")
	fs.Parse(args)

	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading config: %v\n", err)
		os.Exit(1)
	}

	p := *port
	if p == 0 {
		p = cfg.Presence.Port
	}
	if p == 0 {
		p = 3000
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nshutting down...")
		cancel()
	}()

	srv := presence.NewServer(cfg)
	if err := srv.Run(ctx, fmt.Sprintf("localhost:%d", p)); err != nil {
		fmt.Fprintf(os.Stderr, "presence server error: %v\n", err)
		os.Exit(1)
	}
}

func listScenarios() {
	scenarios := orchestrator.ListScenarios()
	fmt.Printf("\nAvailable test scenarios (%d):\n\n", len(scenarios))
	for _, s := range scenarios {
		fmt.Printf("  %s [%s] (%d tasks)\n", s.Name, s.Category, s.TaskCount)
		fmt.Printf("    %s\n", s.Description)
		fmt.Printf("    Hypothesis: %s\n\n", s.Hypothesis)
	}
}

func printUsage() {
	fmt.Println(`Susan -- ISC Consciousness Architecture Experiment

Usage:
  susan run [flags]              Run the experiment
  susan evaluate [flags]         Run blind post-hoc evaluation
  susan export-for-rating [flags] Export blinded responses for human raters
  susan import-ratings [flags]   Import completed human rating CSVs
  susan presence [flags]         Interactive presence mode with voice
  susan scenarios                List available test scenarios
  susan help                     Show this help

Run flags:
  --config PATH           Path to config file (default: config.yaml)
  --scenario NAME         Run a single scenario by name
  --mode MODE             Run mode: control, history_only, feedback_blind, self_referential, random_perturb

Evaluate flags:
  --config PATH           Path to config file (default: config.yaml)
  --dir PATH              Path to experiment output directory (required)
  --output PATH           Output directory for evaluations (default: <dir>/blind_eval)

Export flags:
  --config PATH           Path to config file (default: config.yaml)
  --dir PATH              Path to experiment output directory (required)
  --output PATH           Output CSV path (default: human_ratings.csv)

Import flags:
  --config PATH           Path to config file (default: config.yaml)
  --csv PATH              Path to completed human ratings CSV (required)
  --output PATH           Output JSONL path (default: human_evaluations.jsonl)

Environment:
  ANTHROPIC_API_KEY       API key (when provider=anthropic)
  OPENAI_API_KEY          API key (when provider=openai or cross_evaluation)`)
}
