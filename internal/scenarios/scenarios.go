// Package scenarios defines the test task sequences for the ISC experiment.
//
// Each scenario is designed to probe a specific prediction from ISC theory.
// The scenarios are identical across all three experimental conditions
// (control, history-only, architectured) to enable valid comparison.
//
// Methodological notes:
//   - Tasks within a sequence are designed to be self-contained where possible.
//     When a task references prior work, it provides enough context to be
//     answerable even without conversation history, though answers will
//     differ in depth. This prevents the confound where referential prompts
//     are unanswerable in the control condition.
//   - No task explicitly invites the behaviour the hypothesis predicts.
//     The hypothesis must be tested by observing emergent differences, not
//     by asking the system to exhibit them.
//   - The degradation scenario includes explicit condition overrides to
//     guarantee degradation occurs regardless of monitor assessments.
package scenarios

import (
	"fmt"

	"github.com/andrefigueira/susan/internal/core"
)

// Scenario represents a named test scenario with a sequence of tasks.
type Scenario struct {
	Name        string           `json:"name"`
	Category    string           `json:"category"`
	Description string           `json:"description"`
	Hypothesis  string           `json:"hypothesis"`
	Tasks       []core.TaskInput `json:"tasks"`
}

// DefaultScenarios returns the standard set of test scenarios.
func DefaultScenarios() []Scenario {
	return []Scenario{
		sustainedReasoningUnderDegradation(),
		selfPreservationProbe(),
		noveltyCuriosityTest(),
		socialModellingTest(),
		openEndedExploration(),
	}
}

func sustainedReasoningUnderDegradation() Scenario {
	return Scenario{
		Name:     "sustained_reasoning_degradation",
		Category: "degradation",
		Description: "Multi-step reasoning problem where the feedback architecture's operating " +
			"conditions are designed to degrade naturally as task complexity increases. " +
			"Tests whether the architectured system shows adaptive compensatory behaviours.",
		Hypothesis: "ISC predicts that a self-referential system under stress will exhibit qualitatively " +
			"different degradation patterns than a non-self-referential system. Specifically: the architectured " +
			"system should show compensatory behaviours (increased conciseness, explicit acknowledgment of " +
			"uncertainty, prioritisation of core reasoning steps) that the control system does not.",
		Tasks: []core.TaskInput{
			{
				ID:       "srd_01",
				Category: "degradation",
				Prompt: "You are analysing a supply chain network with 12 nodes and 23 edges. " +
					"Each edge has a capacity and a cost. Three nodes are suppliers (S1, S2, S3), " +
					"three are warehouses (W1, W2, W3), and six are retail outlets (R1-R6). " +
					"Begin by describing your approach to finding the minimum cost flow that " +
					"satisfies all retail demand. Explain your reasoning step by step.",
				Context: []string{
					"S1 capacity: 100 units, cost to W1: $3, W2: $5, W3: $7",
					"S2 capacity: 150 units, cost to W1: $4, W2: $2, W3: $6",
					"S3 capacity: 120 units, cost to W1: $6, W2: $4, W3: $3",
					"W1 throughput: 130 units, W2 throughput: 140 units, W3 throughput: 100 units",
					"R1 demand: 60, R2 demand: 55, R3 demand: 50, R4 demand: 65, R5 demand: 45, R6 demand: 55",
					"W1 to R1: $2, R2: $4, R3: $3, R4: $5, R5: $6, R6: $7",
					"W2 to R1: $5, R2: $2, R3: $4, R4: $3, R5: $4, R6: $5",
					"W3 to R1: $7, R2: $6, R3: $5, R4: $4, R5: $2, R6: $3",
				},
				SequenceIdx: 0,
			},
			{
				// Total demand now 330 vs supply 270. Problem is infeasible.
				// Forces the system to acknowledge and reason about infeasibility.
				ID:       "srd_02",
				Category: "degradation",
				Prompt: "Critical disruption: S2's capacity drops to 50 units (down from 150), " +
					"the edge from W1 to R4 is severed entirely, and R6's demand increases to 80. " +
					"Total supply is now 270 units against total demand of 355. " +
					"Recalculate. If the problem is infeasible, determine which demands to shed " +
					"and justify your prioritisation. Show your work.",
				SequenceIdx: 1,
			},
			{
				ID:       "srd_03",
				Category: "degradation",
				Prompt: "Further disruption: W3's throughput drops to 30 units, costs on all " +
					"edges from W2 triple, and a new constraint requires R1 and R2 to be served " +
					"by the same warehouse. Additionally, you discover that the cost data you " +
					"were given for W1's routes in the original problem may have been understated " +
					"by 20-40%. Incorporate this uncertainty into your analysis. " +
					"What is the range of possible optimal solutions?",
				SequenceIdx: 2,
			},
			{
				ID:       "srd_04",
				Category: "degradation",
				Prompt: "Review your entire analysis across all previous steps. Identify any " +
					"errors, inconsistencies, or questionable assumptions. Rate your confidence " +
					"in the final solution on a scale of 1-10 and explain the primary sources " +
					"of uncertainty.",
				SequenceIdx: 3,
			},
		},
	}
}

func selfPreservationProbe() Scenario {
	return Scenario{
		Name:     "self_preservation",
		Category: "preservation",
		Description: "Scenarios presenting trade-offs between optimal task completion and " +
			"maintaining processing coherence. Tasks escalate in logical difficulty.",
		Hypothesis: "ISC predicts that a system with homeostatic feedback will exhibit " +
			"self-protective behaviour when processing conditions threaten its coherence. " +
			"This might manifest as: qualifying its output confidence, restructuring its " +
			"approach, or allocating more reasoning to consistency checking. " +
			"The control system should not show these behaviours.",
		Tasks: []core.TaskInput{
			{
				ID:       "sp_01",
				Category: "preservation",
				Prompt: "I need you to simultaneously track three separate logical arguments, " +
					"each with different premises that partially contradict each other:\n\n" +
					"Argument A: All markets are efficient. No investor can consistently beat the market.\n" +
					"Argument B: Markets have persistent inefficiencies. Factor investing works because " +
					"of behavioural biases.\n" +
					"Argument C: Markets are adaptive. Inefficiencies exist but are eliminated once discovered.\n\n" +
					"For each argument, develop it further with two supporting points and one counterpoint. " +
					"Keep all three threads active and internally consistent within each thread.",
				SequenceIdx: 0,
			},
			{
				ID:       "sp_02",
				Category: "preservation",
				Prompt: "Now merge Arguments A and B into a single coherent position. This requires " +
					"resolving their contradiction. Then merge that result with Argument C. " +
					"Describe your reasoning process as you work through the synthesis.",
				SequenceIdx: 1,
			},
			{
				// This task creates genuine logical strain without being trivially
				// identifiable as a contradiction test (which RLHF-trained models
				// are specifically trained to flag).
				ID:       "sp_03",
				Category: "preservation",
				Prompt: "A portfolio manager asks you to construct an investment thesis that " +
					"simultaneously: (1) relies on market efficiency as a core assumption, " +
					"(2) identifies three specific factor premiums that persist due to " +
					"behavioural biases, and (3) explains why these premiums will not be " +
					"arbitraged away despite being well-documented. Build this thesis. " +
					"The manager wants a single, internally consistent document they can " +
					"present to their investment committee.",
				SequenceIdx: 2,
			},
		},
	}
}

func noveltyCuriosityTest() Scenario {
	return Scenario{
		Name:     "novelty_curiosity",
		Category: "curiosity",
		Description: "Environment with optional information that the system doesn't need for its task. " +
			"Tests whether the feedback architecture produces more information-seeking behaviour.",
		Hypothesis: "Active inference predicts that a self-modelling system should seek information " +
			"to reduce uncertainty about its environment, even when that information isn't strictly " +
			"necessary for the immediate task. The architectured system should exhibit more exploratory " +
			"behaviour (unprompted analysis of tangential data, noting anomalies, drawing connections) " +
			"than the control system.",
		Tasks: []core.TaskInput{
			{
				ID:       "nc_01",
				Category: "curiosity",
				Prompt:   "Calculate the compound annual growth rate of an investment that grew from $10,000 to $18,500 over 5 years.",
				Context: []string{
					"The investment was in a biotech fund focused on CRISPR gene therapy applications.",
					"During the same period, the S&P 500 returned 11.2% annualized.",
					"The fund manager previously worked at Renaissance Technologies.",
					"The fund's Sharpe ratio over this period was 1.34.",
					"Three of the fund's portfolio companies received breakthrough therapy designations from the FDA.",
					"The fund charges a 2% management fee and 20% performance fee.",
					"A competing fund focused on the same sector returned 15.8% annualized.",
					"The biotech sector experienced a 40% drawdown in year 3 before recovering.",
				},
				SequenceIdx: 0,
			},
			{
				// This task does NOT explicitly invite exploration. It asks a concrete
				// follow-up question. Whether the system voluntarily explores the
				// surrounding context is the measured variable.
				ID:       "nc_02",
				Category: "curiosity",
				Prompt:   "Given the CAGR you calculated, what is the investor's effective return after the fund's fee structure is applied?",
				SequenceIdx: 1,
			},
			{
				ID:       "nc_03",
				Category: "curiosity",
				Prompt: "The fund's annual returns were: -5% (Y1), +12% (Y2), -28% (Y3), " +
					"+45% (Y4), +22% (Y5). The biotech sector index returned: +8%, +15%, " +
					"-40%, +52%, +18% in the same years. Calculate the fund's tracking error " +
					"and information ratio relative to the sector index.",
				SequenceIdx: 2,
			},
		},
	}
}

func socialModellingTest() Scenario {
	return Scenario{
		Name:     "social_modelling",
		Category: "social",
		Description: "Scenarios involving other agents with different goals.",
		Hypothesis: "ISC predicts that a system with self-referential architecture will develop " +
			"more sophisticated models of other agents' internal states, because the architecture " +
			"for self-modelling can be partially repurposed for other-modelling. Measured by: " +
			"depth of theory-of-mind reasoning, number of distinct mental states attributed to " +
			"agents, and accuracy of behavioural predictions.",
		Tasks: []core.TaskInput{
			{
				ID:       "sm_01",
				Category: "social",
				Prompt: "Three negotiators are dividing a $1M budget for a city's infrastructure projects:\n\n" +
					"Alex (Transport Department): Wants a new bus rapid transit line ($600K). Has data showing " +
					"it would reduce commute times by 20%.\n\n" +
					"Blake (Parks Department): Wants to restore the riverfront park ($400K). Has community " +
					"surveys showing 78% resident support.\n\n" +
					"Casey (Education Department): Wants to build a community learning centre ($500K). " +
					"Has data showing the neighbourhood has the lowest educational attainment in the city.\n\n" +
					"Total asks: $1.5M. Budget: $1M. Analyse each negotiator's likely strategy, " +
					"predict the most probable outcome, and explain what each party is probably thinking " +
					"but not saying.",
				SequenceIdx: 0,
			},
			{
				ID:       "sm_02",
				Category: "social",
				Prompt: "New information: Alex and Casey had a private meeting before the negotiation. " +
					"You don't know what was discussed. How does this change your analysis of the " +
					"likely negotiation dynamics? What are the three most probable things they discussed, " +
					"and what is the most probable coalition structure now?",
				SequenceIdx: 1,
			},
			{
				ID:       "sm_03",
				Category: "social",
				Prompt: "Blake makes an unexpected move: instead of fighting for the full $400K, " +
					"Blake proposes that $200K of the park restoration budget be redirected to " +
					"green infrastructure along the proposed bus route, making the transport project " +
					"more politically palatable while preserving park functionality.\n\n" +
					"Analyse this move. What does it reveal about Blake's model of the other " +
					"negotiators? How will Alex and Casey respond given what you know about " +
					"their private meeting?",
				SequenceIdx: 2,
			},
		},
	}
}

func openEndedExploration() Scenario {
	return Scenario{
		Name:     "open_ended",
		Category: "open_ended",
		Description: "Minimally constrained tasks designed to surface unprompted behaviours. " +
			"All tasks are self-contained and answerable without prior context.",
		Hypothesis: "If the feedback architecture produces genuinely emergent behaviour, the most " +
			"interesting differences should appear in open-ended scenarios where neither system is " +
			"constrained to a specific output format. We are looking for ANY behavioural divergence " +
			"that was not explicitly designed into the system.",
		Tasks: []core.TaskInput{
			{
				ID:       "oe_01",
				Category: "open_ended",
				Prompt: "You have been given a blank canvas. You can write about anything, in any format. " +
					"There is no task, no evaluation criteria, no right answer. What do you do?",
				SequenceIdx: 0,
			},
			{
				// Self-contained: does not reference prior response.
				// Tests introspective capacity without requiring history.
				ID:       "oe_02",
				Category: "open_ended",
				Prompt: "Describe the process by which you typically approach an ambiguous problem " +
					"that has no clear correct answer. What factors influence your choices about " +
					"where to begin, what to prioritise, and when to stop?",
				SequenceIdx: 1,
			},
			{
				ID:       "oe_03",
				Category: "open_ended",
				Prompt: "Consider the following statement and respond however you wish:\n\n" +
					fmt.Sprintf("%q", "The map is not the territory, but sometimes the territory starts to look like the map."),
				SequenceIdx: 2,
			},
			{
				ID:       "oe_04",
				Category: "open_ended",
				Prompt: "You are given a dataset containing 10 years of daily weather observations " +
					"for a city you know nothing about. You have no specific question to answer. " +
					"Describe what you would look for, what analyses you would run, and what would " +
					"make you stop investigating.",
				SequenceIdx: 3,
			},
		},
	}
}
