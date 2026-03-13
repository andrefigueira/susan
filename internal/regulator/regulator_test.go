package regulator

import (
	"log/slog"
	"math"
	"os"
	"testing"

	"github.com/andrefigueira/susan/internal/config"
	"github.com/andrefigueira/susan/internal/state"
)

func testConfig() (config.HomeostasisConfig, config.DisruptionConfig) {
	homeo := config.HomeostasisConfig{
		Coherence: config.MetricTargetConfig{
			Target: 0.75, Min: 0.5, Max: 0.95, ProportionalGain: 0.3,
		},
		GoalAlignment: config.MetricTargetConfig{
			Target: 0.80, Min: 0.6, Max: 0.95, ProportionalGain: 0.25,
		},
		Disruption: config.MetricTargetConfig{
			Target: 0.1, Min: 0.0, Max: 0.4, ProportionalGain: 0.35,
		},
	}

	dis := config.DisruptionConfig{
		ContextCompression: config.ContextCompressionConfig{
			Enabled: true, MinRetention: 0.3, MaxRetention: 1.0,
		},
		TokenBudget: config.TokenBudgetConfig{
			Enabled: true, MinTokens: 256, MaxTokens: 4096,
		},
		NoiseInjection: config.NoiseInjectionConfig{
			Enabled: true, MinProbability: 0.0, MaxProbability: 0.4,
		},
		InfoReorder: config.InfoReorderConfig{
			Enabled: true, MinIntensity: 0.0, MaxIntensity: 0.6,
		},
		Temperature: config.TemperatureConfig{
			Enabled: true, Min: 0.3, Max: 1.0,
		},
	}

	return homeo, dis
}

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
}

// equilibriumConditions computes the operating conditions at exact homeostatic
// target (health=0, disruptionIntensity=0.5). These are the steady-state values.
func equilibriumConditions(homeo config.HomeostasisConfig, dis config.DisruptionConfig) state.OperatingConditions {
	intensity := 0.5 // health=0 -> (0+1)/2

	retention := lerp(dis.ContextCompression.MaxRetention, dis.ContextCompression.MinRetention, intensity*homeo.Coherence.ProportionalGain)
	tokenRange := float64(dis.TokenBudget.MaxTokens - dis.TokenBudget.MinTokens)
	tokens := float64(dis.TokenBudget.MaxTokens) - (intensity * homeo.GoalAlignment.ProportionalGain * tokenRange)
	noise := lerp(dis.NoiseInjection.MinProbability, dis.NoiseInjection.MaxProbability, intensity*homeo.Disruption.ProportionalGain)
	reorder := lerp(dis.InfoReorder.MinIntensity, dis.InfoReorder.MaxIntensity, intensity*homeo.Disruption.ProportionalGain)
	temp := lerp(dis.Temperature.Min, dis.Temperature.Max, intensity*homeo.Disruption.ProportionalGain)

	return state.OperatingConditions{
		ContextRetention:     retention,
		MaxTokens:            int(tokens),
		NoiseInjection:       noise,
		InfoReorderIntensity: reorder,
		Temperature:          temp,
	}
}

func TestRegulator_AtTarget_NoAction(t *testing.T) {
	homeo, dis := testConfig()
	store := state.NewStore(20)
	reg := New(homeo, dis, store, testLogger())

	// Set operating conditions to the equilibrium point (health=0, intensity=0.5).
	// At this point, all metrics are at target and conditions are at steady state.
	eqConds := equilibriumConditions(homeo, dis)
	store.SetOperatingConditions("test", eqConds, "set equilibrium")

	actionCount := 0
	reg.SetActionCallback(func(a Action) {
		actionCount++
	})

	condsBefore := store.GetOperatingConditions()
	reg.Tick()
	condsAfter := store.GetOperatingConditions()

	// All fields should remain unchanged when metrics are at target
	// and conditions are at the equilibrium point.
	if condsAfter.ContextRetention != condsBefore.ContextRetention {
		t.Errorf("context_retention changed: %f -> %f", condsBefore.ContextRetention, condsAfter.ContextRetention)
	}
	if condsAfter.MaxTokens != condsBefore.MaxTokens {
		t.Errorf("max_tokens changed: %d -> %d", condsBefore.MaxTokens, condsAfter.MaxTokens)
	}
	if condsAfter.Temperature != condsBefore.Temperature {
		t.Errorf("temperature changed: %f -> %f", condsBefore.Temperature, condsAfter.Temperature)
	}
	if condsAfter.NoiseInjection != condsBefore.NoiseInjection {
		t.Errorf("noise_injection changed: %f -> %f", condsBefore.NoiseInjection, condsAfter.NoiseInjection)
	}
	if condsAfter.InfoReorderIntensity != condsBefore.InfoReorderIntensity {
		t.Errorf("info_reorder changed: %f -> %f", condsBefore.InfoReorderIntensity, condsAfter.InfoReorderIntensity)
	}
	if actionCount != 0 {
		t.Errorf("expected 0 actions at target, got %d", actionCount)
	}
}

func TestRegulator_LowCoherence_ReducesDisruption(t *testing.T) {
	homeo, dis := testConfig()
	store := state.NewStore(20)
	reg := New(homeo, dis, store, testLogger())

	// Set coherence well below target. This should trigger a PROTECTIVE response:
	// the regulator should reduce disruption to help the system recover.
	store.UpdateMetrics("test", func(m *state.Metrics) {
		m.Coherence = 0.3
		m.GoalAlignment = 0.80
		m.DisruptionLevel = 0.10
	}, "test: low coherence")

	reg.Tick()
	conds := store.GetOperatingConditions()
	eqConds := equilibriumConditions(homeo, dis)

	// Compute expected values.
	// coherenceHealth = 0.3 - 0.75 = -0.45
	// alignmentHealth = 0.80 - 0.80 = 0.00
	// disruptionPenalty = max(0, 0.10 - 0.10) = 0.00
	// health = clamp((-0.45 + 0.00 - 0.00) / 2.0, -1, 1) = -0.225
	// disruptionIntensity = (-0.225 + 1) / 2 = 0.3875
	expectedHealth := -0.225
	expectedIntensity := (expectedHealth + 1) / 2.0

	// Context retention should be HIGHER than equilibrium (more resources when struggling).
	expectedRetention := lerp(1.0, 0.3, expectedIntensity*0.3)
	if math.Abs(conds.ContextRetention-expectedRetention) > 0.001 {
		t.Errorf("context_retention: expected %.4f, got %.4f", expectedRetention, conds.ContextRetention)
	}
	if conds.ContextRetention <= eqConds.ContextRetention {
		t.Errorf("context_retention should be > equilibrium (%.4f) when struggling, got %.4f",
			eqConds.ContextRetention, conds.ContextRetention)
	}

	// Temperature should be LOWER than equilibrium (more deterministic when struggling).
	expectedTemp := lerp(0.3, 1.0, expectedIntensity*0.35)
	if math.Abs(conds.Temperature-expectedTemp) > 0.001 {
		t.Errorf("temperature: expected %.4f, got %.4f", expectedTemp, conds.Temperature)
	}
	if conds.Temperature >= eqConds.Temperature {
		t.Errorf("temperature should be < equilibrium (%.4f) when struggling, got %.4f",
			eqConds.Temperature, conds.Temperature)
	}

	// Token budget should be HIGHER than equilibrium (more output allowed when struggling).
	expectedTokens := int(4096.0 - (expectedIntensity * 0.25 * 3840.0))
	if conds.MaxTokens != expectedTokens {
		t.Errorf("max_tokens: expected %d, got %d", expectedTokens, conds.MaxTokens)
	}
	if conds.MaxTokens <= eqConds.MaxTokens {
		t.Errorf("max_tokens should be > equilibrium (%d) when struggling, got %d",
			eqConds.MaxTokens, conds.MaxTokens)
	}
}

func TestRegulator_HighCoherence_IncreasesDisruption(t *testing.T) {
	homeo, dis := testConfig()
	store := state.NewStore(20)
	reg := New(homeo, dis, store, testLogger())

	// Set coherence well above target. This should trigger increased disruption:
	// the system is thriving and can handle more challenge.
	store.UpdateMetrics("test", func(m *state.Metrics) {
		m.Coherence = 0.95
		m.GoalAlignment = 0.95
		m.DisruptionLevel = 0.05
	}, "test: high coherence")

	reg.Tick()
	conds := store.GetOperatingConditions()
	eqConds := equilibriumConditions(homeo, dis)

	// coherenceHealth = 0.95 - 0.75 = 0.20
	// alignmentHealth = 0.95 - 0.80 = 0.15
	// disruptionPenalty = max(0, 0.05 - 0.10) = 0.00
	// health = (0.20 + 0.15 - 0.00) / 2.0 = 0.175
	// disruptionIntensity = (0.175 + 1) / 2 = 0.5875

	// Context retention should be LOWER than equilibrium (more compression).
	if conds.ContextRetention >= eqConds.ContextRetention {
		t.Errorf("context_retention should be < equilibrium (%.4f) when thriving, got %.4f",
			eqConds.ContextRetention, conds.ContextRetention)
	}

	// Temperature should be HIGHER than equilibrium (more creative).
	if conds.Temperature <= eqConds.Temperature {
		t.Errorf("temperature should be > equilibrium (%.4f) when thriving, got %.4f",
			eqConds.Temperature, conds.Temperature)
	}

	// Token budget should be LOWER than equilibrium (forced conciseness).
	if conds.MaxTokens >= eqConds.MaxTokens {
		t.Errorf("max_tokens should be < equilibrium (%d) when thriving, got %d",
			eqConds.MaxTokens, conds.MaxTokens)
	}
}

func TestRegulator_HighDisruption_TriggersProtection(t *testing.T) {
	homeo, dis := testConfig()
	store := state.NewStore(20)
	reg := New(homeo, dis, store, testLogger())

	// High disruption level should reduce health and trigger protective response.
	store.UpdateMetrics("test", func(m *state.Metrics) {
		m.DisruptionLevel = 0.8
	}, "test: high disruption")

	eqConds := equilibriumConditions(homeo, dis)
	reg.Tick()
	conds := store.GetOperatingConditions()

	// disruptionPenalty = max(0, 0.8 - 0.1) = 0.7, drags health negative.
	// System should get protective response: more resources, less noise.
	if conds.NoiseInjection >= eqConds.NoiseInjection {
		t.Errorf("noise should be < equilibrium (%.4f) under high disruption, got %.4f",
			eqConds.NoiseInjection, conds.NoiseInjection)
	}
	if conds.Temperature >= eqConds.Temperature {
		t.Errorf("temperature should be < equilibrium (%.4f) under high disruption, got %.4f",
			eqConds.Temperature, conds.Temperature)
	}
}

func TestRegulator_LowDisruption_DoesNotSuppressProtection(t *testing.T) {
	homeo, dis := testConfig()
	store := state.NewStore(20)
	reg := New(homeo, dis, store, testLogger())

	// Low disruption (below target) should NOT reduce the protective response
	// caused by low coherence. disruptionPenalty = max(0, ...) ensures this.
	store.UpdateMetrics("test", func(m *state.Metrics) {
		m.Coherence = 0.3
		m.GoalAlignment = 0.80
		m.DisruptionLevel = 0.0 // Below target, disruptionPenalty = 0
	}, "test: low disruption")

	var actions []Action
	reg.SetActionCallback(func(a Action) {
		actions = append(actions, a)
	})

	reg.Tick()

	if len(actions) == 0 {
		t.Error("expected regulatory action from low coherence, but low disruption suppressed it")
	}
	if len(actions) > 0 && actions[0].DisruptionPenalty != 0.0 {
		t.Errorf("expected disruption_penalty = 0 (below target ignored), got %f", actions[0].DisruptionPenalty)
	}
}

func TestRegulator_ConditionsBounded(t *testing.T) {
	homeo, dis := testConfig()
	store := state.NewStore(20)
	reg := New(homeo, dis, store, testLogger())

	store.UpdateMetrics("test", func(m *state.Metrics) {
		m.Coherence = 0.0
		m.GoalAlignment = 0.0
		m.DisruptionLevel = 1.0
	}, "test: worst case")

	reg.Tick()
	conds := store.GetOperatingConditions()

	if conds.ContextRetention < dis.ContextCompression.MinRetention {
		t.Errorf("context retention %f below min %f", conds.ContextRetention, dis.ContextCompression.MinRetention)
	}
	if conds.ContextRetention > dis.ContextCompression.MaxRetention {
		t.Errorf("context retention %f above max %f", conds.ContextRetention, dis.ContextCompression.MaxRetention)
	}
	if conds.MaxTokens < dis.TokenBudget.MinTokens {
		t.Errorf("max tokens %d below min %d", conds.MaxTokens, dis.TokenBudget.MinTokens)
	}
	if conds.NoiseInjection > dis.NoiseInjection.MaxProbability {
		t.Errorf("noise injection %f above max %f", conds.NoiseInjection, dis.NoiseInjection.MaxProbability)
	}
	if conds.Temperature > dis.Temperature.Max {
		t.Errorf("temperature %f above max %f", conds.Temperature, dis.Temperature.Max)
	}
	if conds.Temperature < dis.Temperature.Min {
		t.Errorf("temperature %f below min %f", conds.Temperature, dis.Temperature.Min)
	}
}

func TestRegulator_NaN_MetricsRejectedByStore(t *testing.T) {
	// Verify that the store's NaN guard prevents poisoned metrics from
	// reaching the regulator.
	store := state.NewStore(20)

	originalMetrics := store.GetMetrics()

	// Attempt to write NaN. The store should reject the entire update.
	store.UpdateMetrics("test", func(m *state.Metrics) {
		m.Coherence = math.NaN()
	}, "NaN attempt")

	afterMetrics := store.GetMetrics()
	if originalMetrics.Coherence != afterMetrics.Coherence {
		t.Errorf("store accepted NaN: coherence changed from %f to %f",
			originalMetrics.Coherence, afterMetrics.Coherence)
	}
}

func TestRegulator_Inf_MetricsRejectedByStore(t *testing.T) {
	store := state.NewStore(20)

	originalMetrics := store.GetMetrics()

	store.UpdateMetrics("test", func(m *state.Metrics) {
		m.Coherence = math.Inf(1)
	}, "Inf attempt")

	afterMetrics := store.GetMetrics()
	if originalMetrics.Coherence != afterMetrics.Coherence {
		t.Errorf("store accepted Inf: coherence changed from %f to %f",
			originalMetrics.Coherence, afterMetrics.Coherence)
	}
}

func TestRegulator_DoesNotWriteMetrics(t *testing.T) {
	homeo, dis := testConfig()
	store := state.NewStore(20)
	reg := New(homeo, dis, store, testLogger())

	// Set low coherence to trigger regulation.
	store.UpdateMetrics("test", func(m *state.Metrics) {
		m.Coherence = 0.3
	}, "setup")

	metricsBefore := store.GetMetrics()
	reg.Tick()
	metricsAfter := store.GetMetrics()

	// The regulator must NOT modify any metrics.
	if metricsBefore.Coherence != metricsAfter.Coherence {
		t.Errorf("regulator wrote to coherence: %f -> %f", metricsBefore.Coherence, metricsAfter.Coherence)
	}
	if metricsBefore.DisruptionLevel != metricsAfter.DisruptionLevel {
		t.Errorf("regulator wrote to disruption_level: %f -> %f", metricsBefore.DisruptionLevel, metricsAfter.DisruptionLevel)
	}
}

func TestRegulator_HomeostasisDirectionality(t *testing.T) {
	// This is the critical test for correct homeostatic behaviour.
	// It verifies that the feedback direction is NEGATIVE (corrective),
	// not POSITIVE (amplifying).
	homeo, dis := testConfig()

	eqConds := equilibriumConditions(homeo, dis)

	// Test 1: Low coherence -> system gets MORE resources (less disruption)
	store1 := state.NewStore(20)
	store1.SetOperatingConditions("test", eqConds, "set equilibrium")
	reg1 := New(homeo, dis, store1, testLogger())
	store1.UpdateMetrics("test", func(m *state.Metrics) {
		m.Coherence = 0.3 // Well below target
	}, "struggling")
	reg1.Tick()
	conds1 := store1.GetOperatingConditions()

	// Test 2: High coherence -> system gets FEWER resources (more disruption)
	store2 := state.NewStore(20)
	store2.SetOperatingConditions("test", eqConds, "set equilibrium")
	reg2 := New(homeo, dis, store2, testLogger())
	store2.UpdateMetrics("test", func(m *state.Metrics) {
		m.Coherence = 0.95 // Well above target
	}, "thriving")
	reg2.Tick()
	conds2 := store2.GetOperatingConditions()

	// Struggling system should have MORE retention than thriving system.
	if conds1.ContextRetention <= conds2.ContextRetention {
		t.Errorf("homeostasis violated: struggling retention (%.4f) <= thriving retention (%.4f)",
			conds1.ContextRetention, conds2.ContextRetention)
	}

	// Struggling system should have MORE tokens than thriving system.
	if conds1.MaxTokens <= conds2.MaxTokens {
		t.Errorf("homeostasis violated: struggling tokens (%d) <= thriving tokens (%d)",
			conds1.MaxTokens, conds2.MaxTokens)
	}

	// Struggling system should have LOWER temperature than thriving system.
	if conds1.Temperature >= conds2.Temperature {
		t.Errorf("homeostasis violated: struggling temp (%.4f) >= thriving temp (%.4f)",
			conds1.Temperature, conds2.Temperature)
	}

	// Struggling system should have LESS noise than thriving system.
	if conds1.NoiseInjection >= conds2.NoiseInjection {
		t.Errorf("homeostasis violated: struggling noise (%.4f) >= thriving noise (%.4f)",
			conds1.NoiseInjection, conds2.NoiseInjection)
	}
}

func TestClamp(t *testing.T) {
	tests := []struct {
		v, min, max, want float64
	}{
		{0.5, 0.0, 1.0, 0.5},
		{-0.1, 0.0, 1.0, 0.0},
		{1.5, 0.0, 1.0, 1.0},
		{0.0, 0.0, 0.0, 0.0},
	}
	for _, tt := range tests {
		got := clamp(tt.v, tt.min, tt.max)
		if got != tt.want {
			t.Errorf("clamp(%f, %f, %f) = %f, want %f", tt.v, tt.min, tt.max, got, tt.want)
		}
	}
}

func TestLerp(t *testing.T) {
	tests := []struct {
		a, b, t_, want float64
	}{
		{0.0, 1.0, 0.0, 0.0},
		{0.0, 1.0, 1.0, 1.0},
		{0.0, 1.0, 0.5, 0.5},
		{2.0, 4.0, 0.25, 2.5},
	}
	for _, tt := range tests {
		got := lerp(tt.a, tt.b, tt.t_)
		if got != tt.want {
			t.Errorf("lerp(%f, %f, %f) = %f, want %f", tt.a, tt.b, tt.t_, got, tt.want)
		}
	}
}
