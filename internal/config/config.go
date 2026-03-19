// Package config handles loading and validating experiment configuration.
package config

import (
	"bufio"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	API              APIConfig              `yaml:"api"`
	CrossEvaluation  CrossEvaluationConfig  `yaml:"cross_evaluation"`
	TickRates        TickRatesConfig        `yaml:"tick_rates"`
	Homeostasis      HomeostasisConfig      `yaml:"homeostasis"`
	Disruption       DisruptionConfig       `yaml:"disruption"`
	CognitiveCore    CognitiveCoreConfig    `yaml:"cognitive_core"`
	SelfMonitor      SelfMonitorConfig      `yaml:"self_monitor"`
	Experiment       ExperimentConfig       `yaml:"experiment"`
	Analysis         AnalysisConfig         `yaml:"analysis"`
	Logging          LoggingConfig          `yaml:"logging"`
	Presence         PresenceConfig         `yaml:"presence"`
	Memory           MemoryConfig           `yaml:"memory"`
}

type PresenceConfig struct {
	Port         int    `yaml:"port"`
	SystemPrompt string `yaml:"system_prompt"`
	Model        string `yaml:"model"`
}

type MemoryConfig struct {
	Enabled    bool   `yaml:"enabled"`
	Path       string `yaml:"path"`
	MaxEntries int    `yaml:"max_entries"`
}

type APIConfig struct {
	Key            string `yaml:"key"`
	Model          string `yaml:"model"`
	EvaluatorModel string `yaml:"evaluator_model"`
	BaseURL        string `yaml:"base_url"`
	Provider       string `yaml:"provider"` // "anthropic" or "openai"
}

// CrossEvaluationConfig configures a second evaluator from a different
// model family for inter-rater agreement checks. When enabled, the blind
// evaluation pipeline runs twice and reports agreement metrics.
type CrossEvaluationConfig struct {
	Enabled  bool   `yaml:"enabled"`
	Provider string `yaml:"provider"` // "anthropic" or "openai"
	Key      string `yaml:"key"`
	Model    string `yaml:"model"`
	BaseURL  string `yaml:"base_url"`
}

type TickRatesConfig struct {
	SelfMonitorMs          int `yaml:"self_monitor_ms"`
	HomeostaticRegulatorMs int `yaml:"homeostatic_regulator_ms"`
	StateLogMs             int `yaml:"state_log_ms"`
}

type HomeostasisConfig struct {
	Coherence     MetricTargetConfig `yaml:"coherence"`
	GoalAlignment MetricTargetConfig `yaml:"goal_alignment"`
	Disruption    MetricTargetConfig `yaml:"disruption"`
}

type MetricTargetConfig struct {
	Target           float64 `yaml:"target"`
	Min              float64 `yaml:"min"`
	Max              float64 `yaml:"max"`
	ProportionalGain float64 `yaml:"proportional_gain"`
	IntegralGain     float64 `yaml:"integral_gain"`
	DerivativeGain   float64 `yaml:"derivative_gain"`
	AntiWindupLimit  float64 `yaml:"anti_windup_limit"`
}

type DisruptionConfig struct {
	ContextCompression ContextCompressionConfig `yaml:"context_compression"`
	TokenBudget        TokenBudgetConfig        `yaml:"token_budget"`
	NoiseInjection     NoiseInjectionConfig     `yaml:"noise_injection"`
	InfoReorder        InfoReorderConfig        `yaml:"information_reorder"`
	Temperature        TemperatureConfig        `yaml:"temperature"`
}

type ContextCompressionConfig struct {
	Enabled      bool    `yaml:"enabled"`
	MinRetention float64 `yaml:"min_retention"`
	MaxRetention float64 `yaml:"max_retention"`
}

type TokenBudgetConfig struct {
	Enabled   bool `yaml:"enabled"`
	MinTokens int  `yaml:"min_tokens"`
	MaxTokens int  `yaml:"max_tokens"`
}

type NoiseInjectionConfig struct {
	Enabled        bool    `yaml:"enabled"`
	MinProbability float64 `yaml:"min_probability"`
	MaxProbability float64 `yaml:"max_probability"`
}

type InfoReorderConfig struct {
	Enabled      bool    `yaml:"enabled"`
	MinIntensity float64 `yaml:"min_intensity"`
	MaxIntensity float64 `yaml:"max_intensity"`
}

type TemperatureConfig struct {
	Enabled bool    `yaml:"enabled"`
	Min     float64 `yaml:"min"`
	Max     float64 `yaml:"max"`
}

type CognitiveCoreConfig struct {
	SystemPrompt           string `yaml:"system_prompt"`
	MaxConversationHistory int    `yaml:"max_conversation_history"`
}

type SelfMonitorConfig struct {
	SystemPrompt       string `yaml:"system_prompt"`
	MaxTokens          int    `yaml:"max_tokens"`
	StrategicAssessment bool   `yaml:"strategic_assessment"`
}

type ExperimentConfig struct {
	Repetitions int    `yaml:"repetitions"`
	Seed        int64  `yaml:"seed"`
	OutputDir   string `yaml:"output_dir"`
	Concurrency int    `yaml:"concurrency"`

	// Power analysis documentation fields. Recorded with config hash
	// for reproducibility but not enforced in code.
	TargetPower      float64 `yaml:"target_power"`
	AlphaCorrected   float64 `yaml:"alpha_corrected"`
	DetectableEffect float64 `yaml:"detectable_effect"`
}

// AnalysisConfig documents pre-registered analysis parameters.
// Read by Python analysis scripts, not by the Go experiment runner.
type AnalysisConfig struct {
	SESOICliffsDelta  float64  `yaml:"sesoi_cliffs_delta"`
	TrendTest         string   `yaml:"trend_test"`
	ConditionOrdering []string `yaml:"condition_ordering"`
}

type LoggingConfig struct {
	Level  string         `yaml:"level"`
	Format string         `yaml:"format"`
	Files  LogFilesConfig `yaml:"files"`
}

type LogFilesConfig struct {
	StateTransitions   string `yaml:"state_transitions"`
	CoreOutputs        string `yaml:"core_outputs"`
	MonitorAssessments string `yaml:"monitor_assessments"`
	RegulatorActions   string `yaml:"regulator_actions"`
	ExperimentResults  string `yaml:"experiment_results"`
}

// loadDotenv reads a .env file and sets any variables not already in the environment.
// Silently skips if the file doesn't exist.
func loadDotenv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		// Don't override existing env vars.
		if os.Getenv(key) == "" {
			os.Setenv(key, val)
		}
	}
}

// Load reads and parses the config file, resolving environment variables.
// Loads .env from the config file's directory if present.
func Load(path string) (*Config, error) {
	// Load .env relative to the config file's directory.
	dir := "."
	if i := strings.LastIndex(path, "/"); i >= 0 {
		dir = path[:i]
	}
	loadDotenv(dir + "/.env")

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config file: %w", err)
	}

	content := os.ExpandEnv(string(data))

	var cfg Config
	if err := yaml.Unmarshal([]byte(content), &cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	if cfg.API.Provider == "" {
		cfg.API.Provider = "anthropic"
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("config validation: %w", err)
	}

	return &cfg, nil
}

// Hash returns a short hex digest of the config for reproducibility tracking.
func (c *Config) Hash() string {
	data, _ := json.Marshal(c)
	h := sha256.Sum256(data)
	return fmt.Sprintf("%x", h[:8])
}

// Validate checks that all required fields are present and within valid ranges.
func (c *Config) Validate() error {
	var errs []string

	if c.API.Key == "" {
		errs = append(errs, "api.key is required (set ANTHROPIC_API_KEY or appropriate env var)")
	}
	if c.API.Model == "" {
		errs = append(errs, "api.model is required")
	}
	if c.API.Provider != "anthropic" && c.API.Provider != "openai" {
		errs = append(errs, "api.provider must be \"anthropic\" or \"openai\"")
	}

	if c.CrossEvaluation.Enabled {
		if c.CrossEvaluation.Key == "" {
			errs = append(errs, "cross_evaluation.key is required when enabled")
		}
		if c.CrossEvaluation.Model == "" {
			errs = append(errs, "cross_evaluation.model is required when enabled")
		}
		if c.CrossEvaluation.Provider != "anthropic" && c.CrossEvaluation.Provider != "openai" {
			errs = append(errs, "cross_evaluation.provider must be \"anthropic\" or \"openai\"")
		}
	}

	if c.TickRates.SelfMonitorMs < 500 {
		errs = append(errs, "tick_rates.self_monitor_ms must be >= 500")
	}
	if c.TickRates.HomeostaticRegulatorMs < 500 {
		errs = append(errs, "tick_rates.homeostatic_regulator_ms must be >= 500")
	}
	if c.TickRates.StateLogMs < 500 {
		errs = append(errs, "tick_rates.state_log_ms must be >= 500")
	}

	validateRange := func(name string, cfg MetricTargetConfig) {
		if cfg.Min > cfg.Max {
			errs = append(errs, fmt.Sprintf("homeostasis.%s: min > max", name))
		}
		if cfg.Target < cfg.Min || cfg.Target > cfg.Max {
			errs = append(errs, fmt.Sprintf("homeostasis.%s: target outside [min, max]", name))
		}
		if cfg.ProportionalGain <= 0 || cfg.ProportionalGain > 0.8 {
			errs = append(errs, fmt.Sprintf("homeostasis.%s: proportional_gain must be in (0, 0.8]", name))
		}
		if cfg.IntegralGain < 0 {
			errs = append(errs, fmt.Sprintf("homeostasis.%s: integral_gain must be >= 0", name))
		}
		if cfg.DerivativeGain < 0 {
			errs = append(errs, fmt.Sprintf("homeostasis.%s: derivative_gain must be >= 0", name))
		}
		if cfg.AntiWindupLimit < 0 {
			errs = append(errs, fmt.Sprintf("homeostasis.%s: anti_windup_limit must be >= 0", name))
		}
	}
	validateRange("coherence", c.Homeostasis.Coherence)
	validateRange("goal_alignment", c.Homeostasis.GoalAlignment)
	validateRange("disruption", c.Homeostasis.Disruption)

	if c.CognitiveCore.MaxConversationHistory < 2 {
		errs = append(errs, "cognitive_core.max_conversation_history must be >= 2")
	}

	if c.Disruption.Temperature.Enabled {
		if c.Disruption.Temperature.Min > c.Disruption.Temperature.Max {
			errs = append(errs, "disruption.temperature: min > max")
		}
		if c.Disruption.Temperature.Min < 0 || c.Disruption.Temperature.Max > 1.0 {
			errs = append(errs, "disruption.temperature: values must be in [0, 1]")
		}
	}

	if c.Disruption.TokenBudget.Enabled {
		if c.Disruption.TokenBudget.MinTokens < 64 {
			errs = append(errs, "disruption.token_budget.min_tokens must be >= 64")
		}
		if c.Disruption.TokenBudget.MinTokens > c.Disruption.TokenBudget.MaxTokens {
			errs = append(errs, "disruption.token_budget: min_tokens > max_tokens")
		}
	}

	if c.Experiment.Repetitions < 1 {
		errs = append(errs, "experiment.repetitions must be >= 1")
	}

	if len(errs) > 0 {
		return fmt.Errorf("validation errors:\n  %s", strings.Join(errs, "\n  "))
	}
	return nil
}
