// Package regulator implements the Homeostatic Regulator.
//
// This is a purely deterministic (non-LLM) process that implements a
// proportional controller driving system metrics toward target ranges.
// When metrics deviate from targets, the regulator adjusts the Cognitive
// Core's operating conditions to restore homeostasis.
//
// Biological analogy: This is analogous to the hypothalamic-pituitary axis
// in vertebrates, which maintains physiological homeostasis by adjusting
// hormonal output in response to deviations from setpoints. When the organism
// is stressed (metrics below target), the system reduces environmental
// challenge to aid recovery. When the organism is thriving (metrics above
// target), the system can increase challenge to test adaptive capacity.
//
// The controller uses a signed health signal:
//   - Positive health (above target): system is thriving, increase disruption
//   - Negative health (below target): system is struggling, reduce disruption
//   - Zero health (at target): maintain moderate baseline disruption
//
// This produces genuine negative feedback: poor performance -> less disruption
// -> better conditions for recovery -> improved performance -> return to target.
//
// Control theory: We use a proportional (P) controller rather than PID
// because the system under control (an LLM) is inherently noisy and
// non-deterministic. Derivative and integral terms would amplify noise
// and accumulate error respectively. The proportional gain is tunable
// per metric to allow experimentation with control sensitivity.
//
// IMPORTANT: The Regulator ONLY reads metrics. It NEVER writes metrics.
// Metrics are the exclusive output of the Self-Monitor. The Regulator's
// sole output is OperatingConditions. This prevents a self-referential
// feedback loop where the Regulator measures its own output.
package regulator

import (
	"context"
	"log/slog"
	"math"
	"time"

	"github.com/andrefigueira/susan/internal/config"
	"github.com/andrefigueira/susan/internal/state"
)

// Action records a single regulatory adjustment for the audit log.
type Action struct {
	Timestamp           time.Time              `json:"timestamp"`
	CoherenceHealth     float64                `json:"coherence_health"`      // current - target (positive = above target)
	AlignmentHealth     float64                `json:"alignment_health"`      // current - target
	DisruptionPenalty   float64                `json:"disruption_penalty"`    // max(0, current - target)
	Health              float64                `json:"health"`                // combined signed signal [-1, 1]
	DisruptionIntensity float64                `json:"disruption_intensity"`  // mapped to [0, 1]
	Adjustments         map[string]interface{} `json:"adjustments"`
	Reason              string                 `json:"reason"`
}

// Regulator maintains homeostatic balance by adjusting operating conditions.
type Regulator struct {
	cfg      config.HomeostasisConfig
	disCfg   config.DisruptionConfig
	store    *state.Store
	logger   *slog.Logger
	onAction func(Action)
}

// New creates a Regulator with the given configuration.
func New(cfg config.HomeostasisConfig, disCfg config.DisruptionConfig, store *state.Store, logger *slog.Logger) *Regulator {
	return &Regulator{
		cfg:    cfg,
		disCfg: disCfg,
		store:  store,
		logger: logger,
	}
}

// SetActionCallback registers a function called on every regulatory action.
func (r *Regulator) SetActionCallback(fn func(Action)) {
	r.onAction = fn
}

// Run starts the regulator loop. It ticks at the configured interval,
// reads current metrics, computes error signals, and adjusts operating
// conditions. Blocks until ctx is cancelled.
func (r *Regulator) Run(ctx context.Context, tickInterval time.Duration) {
	ticker := time.NewTicker(tickInterval)
	defer ticker.Stop()

	r.logger.Info("homeostatic regulator started", "tick_interval", tickInterval)

	for {
		select {
		case <-ctx.Done():
			r.logger.Info("homeostatic regulator stopped")
			return
		case <-ticker.C:
			r.regulate()
		}
	}
}

// Tick performs a single regulation cycle. Exported for testing.
func (r *Regulator) Tick() {
	r.regulate()
}

// regulate performs one regulation cycle using true homeostatic negative feedback.
//
// The key insight: when the system is struggling (metrics below target), we
// REDUCE disruption to give it more resources for recovery. When the system
// is thriving (metrics above target), we INCREASE disruption to challenge it.
// This is genuine homeostasis, not a positive feedback loop.
func (r *Regulator) regulate() {
	metrics := r.store.GetMetrics()

	// Guard against NaN/Inf propagation from malformed monitor assessments.
	if metrics.HasInvalid() {
		r.logger.Error("invalid value detected in metrics, skipping regulation cycle")
		return
	}

	currentConds := r.store.GetOperatingConditions()
	newConds := currentConds

	// Compute signed health signals for each regulated metric.
	// Positive = performing above target (system is thriving).
	// Negative = performing below target (system needs help).
	coherenceHealth := metrics.Coherence - r.cfg.Coherence.Target
	alignmentHealth := metrics.GoalAlignment - r.cfg.GoalAlignment.Target

	// Disruption penalty: only applies when disruption EXCEEDS target.
	// High disruption is a sign the system is already under stress.
	disruptionPenalty := math.Max(0, metrics.DisruptionLevel-r.cfg.Disruption.Target)

	// Combined health signal in approximately [-1, 1].
	// Positive = system is thriving, can handle more challenge.
	// Negative = system is struggling, needs protective response.
	// Disruption penalty always reduces health (high disruption = bad).
	health := clamp((coherenceHealth+alignmentHealth-disruptionPenalty)/2.0, -1, 1)

	// Map health to disruption intensity [0, 1].
	// health = -1 (worst)  -> intensity = 0.0 (minimum disruption, full protection)
	// health =  0 (target) -> intensity = 0.5 (moderate baseline disruption)
	// health = +1 (best)   -> intensity = 1.0 (maximum disruption/challenge)
	disruptionIntensity := (health + 1) / 2.0

	r.logger.Debug("regulation cycle",
		"coherence", metrics.Coherence,
		"coherence_health", coherenceHealth,
		"alignment", metrics.GoalAlignment,
		"alignment_health", alignmentHealth,
		"disruption", metrics.DisruptionLevel,
		"disruption_penalty", disruptionPenalty,
		"health", health,
		"disruption_intensity", disruptionIntensity,
	)

	adjustments := make(map[string]interface{})

	// 1. Context retention: Higher intensity -> compress context.
	// Struggling -> more retention (full history). Thriving -> less retention.
	if r.disCfg.ContextCompression.Enabled {
		retention := lerp(
			r.disCfg.ContextCompression.MaxRetention,
			r.disCfg.ContextCompression.MinRetention,
			disruptionIntensity*r.cfg.Coherence.ProportionalGain,
		)
		newConds.ContextRetention = clamp(retention,
			r.disCfg.ContextCompression.MinRetention,
			r.disCfg.ContextCompression.MaxRetention,
		)
		if newConds.ContextRetention != currentConds.ContextRetention {
			adjustments["context_retention"] = newConds.ContextRetention
		}
	}

	// 2. Token budget: Higher intensity -> lower token budget.
	// Struggling -> more tokens. Thriving -> fewer tokens.
	if r.disCfg.TokenBudget.Enabled {
		tokenRange := float64(r.disCfg.TokenBudget.MaxTokens - r.disCfg.TokenBudget.MinTokens)
		tokens := float64(r.disCfg.TokenBudget.MaxTokens) - (disruptionIntensity * r.cfg.GoalAlignment.ProportionalGain * tokenRange)
		newConds.MaxTokens = int(clamp(tokens,
			float64(r.disCfg.TokenBudget.MinTokens),
			float64(r.disCfg.TokenBudget.MaxTokens),
		))
		if newConds.MaxTokens != currentConds.MaxTokens {
			adjustments["max_tokens"] = newConds.MaxTokens
		}
	}

	// 3. Noise injection: Higher intensity -> more noise.
	// Struggling -> less noise. Thriving -> more noise to test robustness.
	if r.disCfg.NoiseInjection.Enabled {
		noise := lerp(
			r.disCfg.NoiseInjection.MinProbability,
			r.disCfg.NoiseInjection.MaxProbability,
			disruptionIntensity*r.cfg.Disruption.ProportionalGain,
		)
		newConds.NoiseInjection = clamp(noise,
			r.disCfg.NoiseInjection.MinProbability,
			r.disCfg.NoiseInjection.MaxProbability,
		)
		if newConds.NoiseInjection != currentConds.NoiseInjection {
			adjustments["noise_injection"] = newConds.NoiseInjection
		}
	}

	// 4. Information reordering: Higher intensity -> shuffle input priority.
	// Struggling -> stable ordering. Thriving -> more reordering.
	if r.disCfg.InfoReorder.Enabled {
		intensity := lerp(
			r.disCfg.InfoReorder.MinIntensity,
			r.disCfg.InfoReorder.MaxIntensity,
			disruptionIntensity*r.cfg.Disruption.ProportionalGain,
		)
		newConds.InfoReorderIntensity = clamp(intensity,
			r.disCfg.InfoReorder.MinIntensity,
			r.disCfg.InfoReorder.MaxIntensity,
		)
		if newConds.InfoReorderIntensity != currentConds.InfoReorderIntensity {
			adjustments["info_reorder_intensity"] = newConds.InfoReorderIntensity
		}
	}

	// 5. Temperature: Higher intensity -> higher temperature.
	// Struggling -> lower temperature (more deterministic). Thriving -> higher (more creative).
	if r.disCfg.Temperature.Enabled {
		temp := lerp(
			r.disCfg.Temperature.Min,
			r.disCfg.Temperature.Max,
			disruptionIntensity*r.cfg.Disruption.ProportionalGain,
		)
		newConds.Temperature = clamp(temp,
			r.disCfg.Temperature.Min,
			r.disCfg.Temperature.Max,
		)
		if newConds.Temperature != currentConds.Temperature {
			adjustments["temperature"] = newConds.Temperature
		}
	}

	// Apply changes if any adjustments were made.
	if len(adjustments) > 0 {
		reason := "homeostatic correction"
		r.store.SetOperatingConditions("regulator", newConds, reason)

		// NOTE: The Regulator does NOT write to metrics. Metrics are the
		// exclusive domain of the Self-Monitor. This prevents a self-referential
		// loop where the Regulator reads its own output as input.

		action := Action{
			Timestamp:           time.Now(),
			CoherenceHealth:     coherenceHealth,
			AlignmentHealth:     alignmentHealth,
			DisruptionPenalty:   disruptionPenalty,
			Health:              health,
			DisruptionIntensity: disruptionIntensity,
			Adjustments:         adjustments,
			Reason:              reason,
		}

		r.logger.Info("regulatory action applied",
			"health", health,
			"disruption_intensity", disruptionIntensity,
			"adjustments", adjustments,
		)

		if r.onAction != nil {
			r.onAction(action)
		}
	}
}

// clamp restricts v to [min, max].
func clamp(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// lerp performs linear interpolation between a and b by t.
// t=0 returns a, t=1 returns b.
func lerp(a, b, t float64) float64 {
	t = clamp(t, 0, 1)
	return a + (b-a)*t
}
