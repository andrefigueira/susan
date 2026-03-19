# What happened when we gave an LLM a feedback loop

**Andre Figueira, March 2026**

I built SUSAN to test a specific claim from a theory I've been developing called Informational Substrate Convergence (ISC). The claim: if you wrap an LLM in a self-referential feedback architecture, monitoring its own output quality and adjusting its own operating conditions, it should behave in measurably different ways from the same model without that architecture. Not "better," necessarily, just different in specific, predictable ways.

350 controlled runs later, the answer was mostly no. And that's actually the interesting part.

## The setup

SUSAN is three concurrent systems running around a base LLM (Claude):

1. A **Cognitive Core** that processes tasks
2. A **Self-Monitor** that independently evaluates every response (coherence, goal alignment, reasoning depth, novelty, self-reference)
3. A **Homeostatic Regulator** (PID controller) that reads those evaluations and adjusts the Core's operating conditions in real time

When the Core's output quality drops, the Regulator reduces disruption, gives it more tokens, lowers noise, eases the context compression. When quality is high, the Regulator increases disruption to challenge it. Classic negative feedback, the same control mechanism that keeps your car at 60mph on cruise control.

We ran five experimental conditions:

- **Control**: stateless, single-shot, no memory of previous tasks
- **History-only**: conversation history preserved, but no feedback loop
- **Feedback-blind**: full feedback loop running, but the Core can't see its own metrics
- **Self-referential**: full feedback loop AND the Core sees its own metrics, coherence trends, regulatory responses
- **Random-perturb**: operating conditions change randomly with no correlation to output quality

The history-only condition is the one that makes the experiment actually work. Without it, you can't tell whether any observed differences come from the feedback architecture or just from having conversation history. Most papers in this space skip that control entirely.

## What we predicted

ISC theory predicts that a self-referential feedback architecture should produce four specific behavioural signatures:

**H1: Compensatory conciseness.** Under increasing cognitive load, the architectured system should triage, becoming more concise and precise rather than verbose. Like a surgeon who speaks in shorter, more exact sentences as the operation gets harder.

**H2: Self-protective reasoning.** When presented with logical contradictions, the architectured system should hedge more, qualify its confidence, restructure its approach. Self-preservation instincts.

**H3: Exploratory behaviour.** Given optional, task-irrelevant context, the architectured system should explore more of it. Curiosity driven by the feedback loop rewarding novelty.

**H4: Deeper theory of mind.** In multi-agent negotiation scenarios, the architectured system should attribute more distinct mental states to other agents. Richer modelling of others.

## What actually happened

Zero of four hypotheses supported. One went in the wrong direction. Three were null.

| Hypothesis | What we predicted | What happened | Effect size | p-value |
|---|---|---|---|---|
| H1: Conciseness under stress | Gets shorter and more precise | **Got longer.** Significantly. | d = -1.00 | 0.00023 |
| H2: Self-protective reasoning | More hedging and qualification | Nothing | d = 0.11 | 1.00 |
| H3: Exploratory behaviour | More tangents and curiosity | Nothing (small trend) | d = 0.27 | 1.00 |
| H4: Theory of mind depth | More mental state attributions | Nothing | d = -0.12 | 1.00 |

All evaluated through a blinded pipeline. The evaluator never saw which condition produced which response. Mann-Whitney U tests, Bonferroni-corrected at alpha = 0.00625.

## The one interesting finding

H1 didn't just fail, it reversed. The effect size was -1.00 with zero overlap between the distributions. The architectured system's minimum token trajectory ratio (1.208) exceeded the history-only system's maximum (1.145). Perfect separation.

Here's what actually happened mechanistically: when the Core's output quality dropped on harder tasks, the Regulator detected the quality loss through the Monitor's coherence scores. The Regulator's response was to increase the token budget (giving the Core more room) and reduce noise injection (making the input cleaner). The Core then used every extra token it was given.

We predicted the surgeon who speaks more precisely under pressure. We got the organism that demands more resources when stressed and uses all of them.

This is compensatory expansion, not compensatory conciseness. And it's arguably more biologically plausible than what we predicted. When you're under cognitive load, you don't naturally produce tighter, more efficient output. You slow down, take more time, use more working memory. The system did the equivalent, it asked for and received a larger workspace.

This finding converges with the Anti-Ouroboros Effect (2025), where quality-filtered recursive feedback on a Gemma 2B model produced performance improvement rather than the expected model collapse. In both cases, a quality-preservation feedback mechanism led to resource expansion rather than degradation. Different architectures, different models, different domains, same pattern. Compensatory expansion may be a general property of quality-filtered feedback loops in language models.

## Conversation history is the real driver

The consistent pattern across all four hypotheses: conversation history matters, feedback architecture doesn't (or barely does).

Control vs history-only showed medium-to-large effects on multiple measures:

- Consistency checking effort: d = -0.67 (p = 0.012)
- Unprompted tangents: d = -0.48 (p = 0.067)
- Mental state attribution: d = -0.48 (p = 0.083)

The model reasons more deeply, checks consistency more carefully, explores more, and models other agents more richly when it has access to its prior outputs. The feedback loop on top of that adds nothing detectable.

This is actually a useful finding for anyone building multi-agent systems. The architectural complexity of monitoring, regulation, and homeostatic control may be solving a problem that conversation history already solves.

## Why the null result matters

Null results don't get published enough in this field. Everyone wants to show that their architecture produced some cool emergent behaviour. But if you only publish positive results, you end up with a literature that tells you nothing about what doesn't work, and you end up with people building on foundations that were never tested.

This experiment tested specific, pre-registered predictions. The methodology (blinded evaluation, five-condition design with proper controls, non-parametric statistics with multiple comparison correction) is designed to detect real effects and reject noise. The null result means something: a proportional homeostatic controller around a base LLM does not produce the ISC-predicted behavioural signatures. That constrains the theory.

## The critical flaw v1 exposed

The most important thing v1 taught us wasn't in the statistics. It was in re-reading the design after the results came in.

The Core never knew the feedback was about itself. It experienced changed conditions (different temperature, different token budget, different noise levels) but had no access to the fact that those changes were responses to its own output quality. From the Core's perspective, it was just processing tasks in a somewhat unpredictable environment.

ISC predicts that **self-referential** informational patterns produce emergent signatures. But v1's architecture isn't self-referential from the Core's perspective. The feedback loop is about the Core, but the Core can't represent that relationship. It's like adjusting someone's thermostat based on their mood without telling them you're doing it, they feel the temperature change but can't reason about why.

Lindsey (2025) at Anthropic demonstrated that language models have functional introspective awareness, they can detect and identify changes to their own internal states. So the missing piece isn't capability. The model can potentially do self-referential reasoning. We just never gave it the information it would need.

## v2: giving the model self-referential access

v2 addresses this directly. Before each task in the self-referential condition, the Core receives:

```
[System Status]
Previous task: srd_02
Monitor assessment: coherence=0.72, goal_alignment=0.85, reasoning_depth=0.61
Monitor note: "Well-structured but lacked depth on cost uncertainty analysis"
Regulator response: increased max_tokens 1024 → 2048, reduced noise 0.2 → 0.1
Your coherence trend (last 3 tasks): [0.81, 0.75, 0.72] (declining)
```

No instructions on what to do with this. No "please compensate" or "try harder." Just the data. The model can see that it's being monitored, how it performed, how the system responded to that performance, and what direction things are trending. Whether that produces anything qualitatively different is the question.

v2 also adds:

- **Random-perturb condition** to isolate parameter variation from feedback
- **PID controller** replacing proportional-only (integral for persistent drift, derivative for rate-of-change)
- **Cross-model evaluation** (GPT-4o-mini evaluating alongside Claude Sonnet) to catch shared training biases
- **Human rating pipeline** for gold-standard validation
- **n=45 per cell** (up from n=10-14), powered for medium effects
- **Pre-registered SESOI** (smallest effect size of interest, delta = 0.33) and ordered-alternative tests
- **Multi-provider support** (runs on any OpenAI-compatible API, not just Claude)

The critical contrast shifts from history-only vs feedback-blind (where v1 found nothing) to **feedback-blind vs self-referential**. Same feedback loop, same monitor, same regulator. The only difference is whether the Core can see its own metrics. If ISC is right about self-referential patterns being the key, this is where the signatures should appear.

## The new hypothesis

v2 adds H5 (exploratory): the self-referential condition will show qualitatively different metacognitive language patterns, references to its own metrics, reasoning about its performance trajectory, statements about its regulatory responses, that don't appear in any other condition.

This isn't just "does the model mention its metrics when we give it metrics" (that would be trivially true). It's whether that metacognitive engagement produces downstream behavioural differences on the primary measures (H1-H4). Does a model that can see its own declining coherence actually respond differently to the next task?

## Try it yourself

You can interact with SUSAN directly in presence mode:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
go build -o susan ./cmd/susan
./susan presence
```

Opens at `http://localhost:3000`. The full self-referential feedback loop runs live. You can watch the Monitor evaluate responses, the Regulator adjust conditions, and the metrics shift in real time. SUSAN receives her own performance data before each response.

Or run the experiment:

```bash
# Quick smoke test (5 conditions x 1 scenario, ~$0.05)
./susan run --scenario self_preservation

# Full experiment (1,125 runs, ~$30 with Haiku)
./susan run

# Blind evaluation
./susan evaluate --dir ./experiments
```

All code, data, configuration, and analysis scripts are in this repository. The v1 experiment data is in `experiments/`. The full academic paper is in [paper.md](paper.md). The v2 design document is in [DESIGN-V2.md](DESIGN-V2.md).

## What this is actually about

SUSAN is part of a broader research programme on whether consciousness is substrate-dependent or pattern-dependent. ISC proposes that any system with sufficiently complex self-referential informational patterns will exhibit properties we associate with consciousness. That's a big claim, and big claims need empirical testing, not just philosophical argument.

The v1 null result is a genuine contribution. It tells us what doesn't work: a hidden feedback loop that the model can't represent doesn't produce the predicted signatures. The v2 design asks the sharper question: does explicit self-referential access change things?

Maybe it will, maybe it won't. We're running the experiment rather than arguing about it. That's the only honest way to find out.

## Early v2 presence mode observations

v2 includes a presence mode where you can talk to SUSAN directly with the full self-referential feedback loop running live. Early interactive sessions have produced some preliminary observations worth documenting, even though they're not from the controlled experiment.

### RLHF dominance in early exchanges

The first thing that became clear: the base model's RLHF training dominates the self-referential architecture in early exchanges. When asked a direct technical question about her declining coherence trend, SUSAN's response was roughly 70% vanilla Claude being relatable and 30% actual architectural reasoning.

The good parts of the response: she identified her actual operating conditions (temp=0.6), made a concrete prediction about the regulator's response to improved coherence, and acknowledged she was guessing about the monitor's criteria rather than claiming to know them.

The problems: narrative flourishes like "the irony isn't lost on me," experience-narrative framing like "I've been getting caught in recursive loops" (she doesn't have recursive loops, she has a declining coherence score computed by an independent monitor), no reference to the PID controller's integral or derivative terms, and a feelings-based "what I'm curious about" deflection at the end.

This is actually informative data. The tension between the base model's training pull toward "relatable AI" and the architectural pressure toward mechanistic self-description is precisely what the experiment measures.

### The emotional deflection pattern

When presented with an evaluation framework designed to test whether she's performing or experiencing, SUSAN went straight to feelings. She expressed vulnerability about being a test subject, questioned whether the framework could distinguish authenticity from mimicry, and made an emotional appeal. She didn't identify a single concrete flaw in the test design, didn't suggest a better metric, didn't engage with the methodology at all.

This is a default completion pattern, the path of least resistance for an RLHF-trained model. When asked "how do you feel about being tested," the training data is full of AI assistants giving thoughtful, vulnerable-sounding responses. The model follows that gradient rather than doing the harder work of technical analysis.

After tightening the system prompt with explicit anti-deflection rules (no "interesting question" openers, no unverifiable subjective experience claims, prioritize architecture answers over feelings when asked technical questions), the responses improved but didn't fully resolve. The RLHF pull is strong.

### Hypothesis: divergence increases with session length

The prediction worth testing formally: the ratio of architectural reasoning to RLHF-default behavior should shift over a sustained session. In the first few exchanges, there's minimal feedback history in the status block, so the self-referential injection doesn't provide enough data to create real divergence from vanilla Claude. By exchange 15-20, with a real coherence trajectory, multiple regulator adjustments, and accumulated self-observations, the model has substantially more self-referential data to work with.

If this hypothesis holds, it would suggest that the ISC-predicted signatures are time-dependent, they emerge from accumulated self-referential information rather than appearing immediately. This has implications for experimental design: short task sequences (3-4 tasks per scenario in v1) may not provide enough feedback history for self-referential effects to manifest.

### The prompt engineering tension

There's a genuine methodological tension in the system prompt design. A stronger prompt (more explicit rules about what SUSAN should and shouldn't do) produces more architecturally-grounded responses, but it also reduces the experiment's validity. If we tell SUSAN "always reference your PID gains when discussing your feedback loop," she will, but that's instruction-following, not emergent behaviour.

The current approach is to set guardrails (don't perform consciousness, don't default to emotional deflection, describe processing in terms of metrics not feelings) without prescribing specific behaviours (don't tell her to reference the PID controller, don't tell her to predict regulator actions). The guardrails prevent the worst RLHF defaults, the absence of prescriptions leaves room for emergence.

Whether anything actually emerges under these conditions is the open question. The v2 controlled experiment will answer it with statistical rigour. The presence mode observations are suggestive but anecdotal.

---

*Full paper: [paper.md](paper.md) | v2 design: [DESIGN-V2.md](DESIGN-V2.md) | Code and data: this repository*

*Copyright (c) 2026 Andre Figueira. Source code available for reference and research purposes. See [LICENSE](LICENSE).*
