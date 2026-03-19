package regulator

import (
	"math"
	"testing"
)

// TestPID_POnlyMatchesProportional verifies that when iGain=0 and dGain=0
// the output equals pGain * error, matching the pre-PID proportional controller.
func TestPID_POnlyMatchesProportional(t *testing.T) {
	pid := NewPIDController()

	tests := []struct {
		name  string
		err   float64
		pGain float64
	}{
		{"positive error", 0.25, 0.3},
		{"negative error", -0.45, 0.3},
		{"zero error", 0.0, 0.3},
		{"large gain", 0.5, 0.8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pid.Reset()
			got := pid.Compute("m", tt.err, tt.pGain, 0, 0, 0)
			want := tt.pGain * tt.err
			if math.Abs(got-want) > 1e-12 {
				t.Errorf("Compute(err=%.3f, pGain=%.3f, i=0, d=0) = %.6f, want %.6f",
					tt.err, tt.pGain, got, want)
			}
		})
	}
}

// TestPID_IntegralAccumulates verifies the integral term accumulates across ticks.
func TestPID_IntegralAccumulates(t *testing.T) {
	pid := NewPIDController()

	// Feed a constant error over three ticks and verify the integral grows.
	// With iGain=0.1 and constant error=0.5:
	//   tick 1: integral = 0.5,          iTerm = 0.1 * 0.5 = 0.05
	//   tick 2: integral = 1.0,          iTerm = 0.1 * 1.0 = 0.10
	//   tick 3: integral = 1.5,          iTerm = 0.1 * 1.5 = 0.15
	// pGain=0 so the full output is just the integral term.
	constErr := 0.5
	iGain := 0.1

	for tick := 1; tick <= 3; tick++ {
		out := pid.Compute("m", constErr, 0, iGain, 0, 0)
		expectedITerm := iGain * constErr * float64(tick)
		if math.Abs(out-expectedITerm) > 1e-10 {
			t.Errorf("tick %d: expected iTerm=%.4f, got output=%.4f", tick, expectedITerm, out)
		}
	}
}

// TestPID_AntiWindupClamps verifies the integral accumulator is clamped.
func TestPID_AntiWindupClamps(t *testing.T) {
	pid := NewPIDController()

	// Feed large positive error with anti-windup limit of 1.0.
	// After enough ticks the accumulator should clamp at 1.0.
	limit := 1.0
	iGain := 0.5

	// 10 ticks of error=1.0 would push accumulator to 10 without clamping.
	for i := 0; i < 10; i++ {
		pid.Compute("m", 1.0, 0, iGain, 0, limit)
	}

	// Output should be iGain * limit, not iGain * 10.
	out := pid.Compute("m", 1.0, 0, iGain, 0, limit)
	maxExpected := iGain*limit + 1e-10 // iGain * (limit + 1 error tick) still clamped
	if out > iGain*limit+iGain {
		t.Errorf("anti-windup failed: output %.4f exceeds iGain*limit + iGain = %.4f",
			out, iGain*limit+iGain)
	}
	_ = maxExpected

	// Accumulator should be clamped at limit regardless of how many ticks.
	// Verify by checking that the integral contribution equals iGain * limit.
	// (pGain=0, dGain=0, so output = iGain * clamped_accumulator)
	clampedOut := pid.Compute("m", 0.0, 0, iGain, 0, limit) // error=0, no new accumulation
	// After clamping, accum=limit. Adding error=0: accum remains limit.
	if math.Abs(clampedOut-iGain*limit) > 1e-10 {
		t.Errorf("clamped integral: expected %.4f, got %.4f", iGain*limit, clampedOut)
	}
}

// TestPID_AntiWindupNegative verifies the clamp also applies to negative accumulation.
func TestPID_AntiWindupNegative(t *testing.T) {
	pid := NewPIDController()

	limit := 0.5
	iGain := 1.0

	// Drive accumulator deeply negative.
	for i := 0; i < 20; i++ {
		pid.Compute("m", -1.0, 0, iGain, 0, limit)
	}

	out := pid.Compute("m", 0.0, 0, iGain, 0, limit)
	// Accumulator should be clamped at -limit.
	if math.Abs(out-(-iGain*limit)) > 1e-10 {
		t.Errorf("negative anti-windup: expected %.4f, got %.4f", -iGain*limit, out)
	}
}

// TestPID_DerivativeRespondsToRateOfChange verifies the derivative term fires
// when error changes between ticks and is zero when error is constant.
func TestPID_DerivativeRespondsToRateOfChange(t *testing.T) {
	pid := NewPIDController()
	dGain := 0.2

	// Tick 1: first tick, no derivative (no previous error).
	out1 := pid.Compute("m", 0.5, 0, 0, dGain, 0)
	if out1 != 0.0 {
		t.Errorf("tick 1 (first tick): expected no derivative term, got %.6f", out1)
	}

	// Tick 2: error jumps from 0.5 to 0.8, delta = 0.3.
	out2 := pid.Compute("m", 0.8, 0, 0, dGain, 0)
	expectedD := dGain * (0.8 - 0.5) // 0.2 * 0.3 = 0.06
	if math.Abs(out2-expectedD) > 1e-12 {
		t.Errorf("tick 2 (increasing error): expected dTerm=%.4f, got %.4f", expectedD, out2)
	}

	// Tick 3: same error as tick 2 (no change), derivative should be zero.
	out3 := pid.Compute("m", 0.8, 0, 0, dGain, 0)
	if math.Abs(out3) > 1e-12 {
		t.Errorf("tick 3 (constant error): expected dTerm=0, got %.4f", out3)
	}

	// Tick 4: error decreases from 0.8 to 0.4, delta = -0.4.
	out4 := pid.Compute("m", 0.4, 0, 0, dGain, 0)
	expectedD4 := dGain * (0.4 - 0.8) // 0.2 * -0.4 = -0.08
	if math.Abs(out4-expectedD4) > 1e-12 {
		t.Errorf("tick 4 (decreasing error): expected dTerm=%.4f, got %.4f", expectedD4, out4)
	}
}

// TestPID_ResetClearsState verifies that Reset() removes all accumulated state.
func TestPID_ResetClearsState(t *testing.T) {
	pid := NewPIDController()

	// Accumulate several ticks worth of integral.
	for i := 0; i < 5; i++ {
		pid.Compute("coherence", 0.5, 0.3, 0.1, 0.05, 0)
	}

	// After reset, the next call should behave as if it were the first tick:
	// no integral history, no derivative (first tick flag re-set).
	pid.Reset()

	// On the first post-reset tick: integral accumulator = 0 + error = 0.5.
	// Derivative: first tick, so dTerm = 0.
	err := 0.5
	pGain, iGain, dGain := 0.3, 0.1, 0.05
	out := pid.Compute("coherence", err, pGain, iGain, dGain, 0)
	expectedAfterReset := pGain*err + iGain*err // dTerm = 0 on first tick
	if math.Abs(out-expectedAfterReset) > 1e-12 {
		t.Errorf("after Reset: expected %.6f, got %.6f", expectedAfterReset, out)
	}
}

// TestPID_IndependentMetricState verifies separate metrics do not share accumulators.
func TestPID_IndependentMetricState(t *testing.T) {
	pid := NewPIDController()

	iGain := 0.1

	// Drive "coherence" integral up over 3 ticks.
	for i := 0; i < 3; i++ {
		pid.Compute("coherence", 1.0, 0, iGain, 0, 0)
	}

	// "alignment" should start fresh with no accumulated integral.
	outAlignment := pid.Compute("alignment", 1.0, 0, iGain, 0, 0)
	expectedFirstTick := iGain * 1.0 // accum = 1.0 on first tick
	if math.Abs(outAlignment-expectedFirstTick) > 1e-12 {
		t.Errorf("alignment metric inherited coherence state: expected %.4f, got %.4f",
			expectedFirstTick, outAlignment)
	}

	// "coherence" should now have accum = 4.0 after one more tick.
	outCoherence := pid.Compute("coherence", 1.0, 0, iGain, 0, 0)
	expectedFourthTick := iGain * 4.0
	if math.Abs(outCoherence-expectedFourthTick) > 1e-12 {
		t.Errorf("coherence accum: expected %.4f (4 ticks), got %.4f", expectedFourthTick, outCoherence)
	}
}

// TestPID_CombinedPID verifies a full P+I+D computation on known inputs.
func TestPID_CombinedPID(t *testing.T) {
	pid := NewPIDController()

	pGain, iGain, dGain := 0.3, 0.05, 0.02

	// Tick 1: err=0.4
	//   pTerm = 0.3 * 0.4 = 0.12
	//   integral after = 0.4, iTerm = 0.05 * 0.4 = 0.02
	//   dTerm = 0 (first tick)
	//   total = 0.12 + 0.02 = 0.14
	out1 := pid.Compute("m", 0.4, pGain, iGain, dGain, 0)
	want1 := 0.3*0.4 + 0.05*0.4
	if math.Abs(out1-want1) > 1e-12 {
		t.Errorf("tick 1: expected %.6f, got %.6f", want1, out1)
	}

	// Tick 2: err=0.6
	//   pTerm = 0.3 * 0.6 = 0.18
	//   integral after = 0.4 + 0.6 = 1.0, iTerm = 0.05 * 1.0 = 0.05
	//   dTerm = 0.02 * (0.6 - 0.4) = 0.02 * 0.2 = 0.004
	//   total = 0.18 + 0.05 + 0.004 = 0.234
	out2 := pid.Compute("m", 0.6, pGain, iGain, dGain, 0)
	want2 := 0.3*0.6 + 0.05*1.0 + 0.02*0.2
	if math.Abs(out2-want2) > 1e-12 {
		t.Errorf("tick 2: expected %.6f, got %.6f", want2, out2)
	}
}
