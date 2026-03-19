// Package regulator - PID controller for homeostatic regulation.
//
// The PID controller augments the existing proportional (P) controller with
// optional integral (I) and derivative (D) terms. When iGain=0 and dGain=0
// the output is mathematically identical to pure proportional control,
// preserving backward compatibility.
//
// Integral term: accumulates persistent error over time, correcting slow
// systematic drifts that proportional gain alone cannot eliminate.
//
// Derivative term: responds to the rate of change in error, providing
// anticipatory damping that reduces overshoot in fast-moving signals.
//
// Anti-windup: clamps the integral accumulator to prevent runaway
// accumulation when the system is saturated (e.g., disruption already
// at its minimum and can go no lower).
//
// Each metric has independent state so that coherence and goal_alignment
// accumulate their own histories without cross-contamination.
package regulator

// metricState holds the per-metric PID accumulator state.
type metricState struct {
	integralAccum float64 // running integral of error over ticks
	prevError     float64 // error value from the previous tick (for derivative)
	firstTick     bool    // true until the first Compute call (no derivative on tick 1)
}

// PIDController is a multi-metric proportional-integral-derivative controller.
// It is not safe for concurrent use; the Regulator calls it from a single
// goroutine during regulation cycles.
type PIDController struct {
	state map[string]*metricState
}

// NewPIDController returns an initialised PIDController with no accumulated state.
func NewPIDController() *PIDController {
	return &PIDController{
		state: make(map[string]*metricState),
	}
}

// Compute returns the PID correction signal for the given metric.
//
// Parameters:
//   - metricName: identifies which per-metric state bucket to use
//   - currentError: signed deviation (current - target); positive = above target
//   - pGain: proportional gain (mandatory; must be > 0 per config validation)
//   - iGain: integral gain (0 = disabled)
//   - dGain: derivative gain (0 = disabled)
//   - antiWindupLimit: clamp magnitude for integral accumulator (0 = no clamp)
//
// When iGain=0 and dGain=0 the return value equals pGain * currentError,
// which is identical to the original proportional-only implementation.
func (p *PIDController) Compute(
	metricName string,
	currentError float64,
	pGain, iGain, dGain, antiWindupLimit float64,
) float64 {
	s, ok := p.state[metricName]
	if !ok {
		s = &metricState{firstTick: true}
		p.state[metricName] = s
	}

	// Proportional term.
	pTerm := pGain * currentError

	// Integral term: accumulate then optionally clamp (anti-windup).
	var iTerm float64
	if iGain != 0 {
		s.integralAccum += currentError
		if antiWindupLimit > 0 {
			if s.integralAccum > antiWindupLimit {
				s.integralAccum = antiWindupLimit
			} else if s.integralAccum < -antiWindupLimit {
				s.integralAccum = -antiWindupLimit
			}
		}
		iTerm = iGain * s.integralAccum
	}

	// Derivative term: rate of change of error.
	// Skipped on the first tick because there is no previous error yet.
	var dTerm float64
	if dGain != 0 && !s.firstTick {
		dTerm = dGain * (currentError - s.prevError)
	}

	s.prevError = currentError
	s.firstTick = false

	return pTerm + iTerm + dTerm
}

// Reset clears all accumulated state across every metric.
// Call this between experiment repetitions so that integral history from
// one run does not bleed into the next.
func (p *PIDController) Reset() {
	p.state = make(map[string]*metricState)
}
