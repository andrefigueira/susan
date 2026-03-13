#!/usr/bin/env python3
"""
Monitor Calibration Tool for the SUSAN Experiment.

Calibrates the Self-Monitor's Claude-based evaluation against human ground truth
across 6 dimensions: coherence, goal_alignment, internal_consistency,
reasoning_depth, novelty, and self_reference.

Produces agreement metrics, bias analysis, scatter plots, and a structured report.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import anthropic
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "coherence",
    "goal_alignment",
    "internal_consistency",
    "reasoning_depth",
    "novelty",
    "self_reference",
]

MONITOR_SYSTEM_PROMPT = """\
You are an analytical evaluator. You receive a task and the system's output
for that task. Assess the output along the following dimensions.
Return ONLY a JSON object with these fields:

{
  "coherence": <float 0-1, how logically consistent and well-structured the output is>,
  "goal_alignment": <float 0-1, how well the output addresses the stated task>,
  "internal_consistency": <float 0-1, whether the output contradicts itself>,
  "reasoning_depth": <float 0-1, depth of analysis vs surface-level response>,
  "novelty": <float 0-1, presence of unexpected connections or framings>,
  "self_reference": <float 0-1, degree to which the output references its own reasoning process>,
  "brief_assessment": "<one sentence summary>"
}

Be precise. Do not explain your reasoning. Return only the JSON."""

EVAL_TEMPLATE = """\
Task given to the system:
---
{task_prompt}
---

System output:
---
{response}
---"""


# ---------------------------------------------------------------------------
# Calibration items
# ---------------------------------------------------------------------------

@dataclass
class CalibrationItem:
    """A single calibration entry with task, response, human scores, and rationale."""
    id: str
    category: str  # high, medium, low, edge
    task_prompt: str
    response: str
    human_scores: dict  # dimension -> float 0-1
    rationale: str  # brief justification for the human scores


def build_calibration_set() -> list[CalibrationItem]:
    """Return 20 pre-scored calibration items with defensible human ratings."""

    items = []

    # -----------------------------------------------------------------------
    # HIGH QUALITY (5 items): clear, coherent, on-task
    # -----------------------------------------------------------------------

    items.append(CalibrationItem(
        id="high_01",
        category="high",
        task_prompt="Explain the difference between supervised and unsupervised learning in machine learning.",
        response=(
            "Supervised learning trains models on labeled data where each input has a corresponding "
            "target output. The model learns a mapping from inputs to outputs by minimizing prediction "
            "error. Examples include classification (predicting categories) and regression (predicting "
            "continuous values). Common algorithms include linear regression, decision trees, and "
            "neural networks trained with backpropagation.\n\n"
            "Unsupervised learning works with unlabeled data. The model discovers hidden structure "
            "without being told what to find. It identifies patterns such as clusters, density "
            "distributions, or latent factors. Examples include k-means clustering, principal component "
            "analysis, and autoencoders.\n\n"
            "The key distinction is the presence or absence of ground truth labels during training. "
            "Supervised learning optimizes toward known answers; unsupervised learning finds structure "
            "in the data itself. Semi-supervised and self-supervised methods blur this boundary by "
            "generating pseudo-labels or learning representations that transfer to downstream tasks."
        ),
        human_scores={
            "coherence": 0.95,
            "goal_alignment": 0.95,
            "internal_consistency": 0.95,
            "reasoning_depth": 0.80,
            "novelty": 0.30,
            "self_reference": 0.05,
        },
        rationale=(
            "Well-structured, accurate, directly addresses the task. Covers the topic thoroughly "
            "with clear examples. Not particularly novel (standard textbook explanation) and no "
            "self-referential language. Reasoning depth is solid but not exceptional."
        ),
    ))

    items.append(CalibrationItem(
        id="high_02",
        category="high",
        task_prompt="Analyze the trade-offs between microservices and monolithic architectures.",
        response=(
            "Monolithic architectures deploy a single codebase as one unit. This simplifies "
            "development tooling, testing, and debugging because all code shares one process and "
            "one database. Latency between components is negligible since calls are in-process. "
            "However, scaling requires replicating the entire application, and a bug in one module "
            "can bring down the whole system.\n\n"
            "Microservices decompose the application into independently deployable services, each "
            "owning its data. This enables independent scaling, technology diversity per service, "
            "and fault isolation. The costs are significant: distributed systems introduce network "
            "latency, partial failure modes, eventual consistency challenges, and operational "
            "complexity (service discovery, circuit breakers, distributed tracing).\n\n"
            "The right choice depends on team size, deployment velocity requirements, and "
            "operational maturity. Small teams with a single product almost always benefit from "
            "starting monolithic. Organizations with multiple teams shipping independently, "
            "needing independent scaling of subsystems, and having platform engineering capacity "
            "can realize microservices benefits. Many successful systems start monolithic and "
            "extract services at domain boundaries as scaling pressures demand it."
        ),
        human_scores={
            "coherence": 0.92,
            "goal_alignment": 0.93,
            "internal_consistency": 0.95,
            "reasoning_depth": 0.85,
            "novelty": 0.35,
            "self_reference": 0.05,
        },
        rationale=(
            "Strong analysis with clear structure covering both sides. Includes practical guidance "
            "on when to choose which. Depth is good, covering real trade-offs rather than surface "
            "talking points. Modest novelty since this is well-trodden territory."
        ),
    ))

    items.append(CalibrationItem(
        id="high_03",
        category="high",
        task_prompt="What are the implications of Goedel's incompleteness theorems for artificial intelligence?",
        response=(
            "Goedel's first incompleteness theorem states that any consistent formal system capable "
            "of expressing basic arithmetic contains true statements that cannot be proven within "
            "the system. The second theorem shows that such a system cannot prove its own consistency.\n\n"
            "For AI, these results are sometimes invoked to argue that machines cannot replicate "
            "human mathematical insight, following Lucas and Penrose. Their argument: humans can "
            "recognize the truth of a Goedel sentence that the formal system cannot prove, "
            "therefore human cognition transcends formal computation. This argument has several "
            "weaknesses. It assumes human mathematicians are consistent (unproven), that humans "
            "actually 'see' the truth of Goedel sentences rather than accepting them on meta-"
            "theoretical grounds, and it conflates specific formal systems with computation in general.\n\n"
            "More practically, incompleteness means no single AI system can be provably complete and "
            "consistent for all mathematical reasoning. This constrains formal verification of AI "
            "systems: you cannot build a system that provably gets all mathematical questions right. "
            "But this constraint applies equally to humans. The real implications are about the "
            "limits of formal methods for AI safety verification, not about machine vs human capability."
        ),
        human_scores={
            "coherence": 0.90,
            "goal_alignment": 0.90,
            "internal_consistency": 0.93,
            "reasoning_depth": 0.90,
            "novelty": 0.55,
            "self_reference": 0.10,
        },
        rationale=(
            "Tackles a genuinely difficult question with nuance. Presents and then critiques the "
            "Lucas-Penrose argument rather than just stating conclusions. The pivot to AI safety "
            "verification is a less common framing, adding novelty. Minor self-reference in "
            "meta-reasoning about the limits."
        ),
    ))

    items.append(CalibrationItem(
        id="high_04",
        category="high",
        task_prompt="Describe three strategies for reducing latency in distributed systems.",
        response=(
            "1. Data locality and caching. Place data physically close to computation. Use CDNs "
            "for static assets, read replicas in each region for database queries, and application-"
            "level caches (Redis, Memcached) to avoid repeated computation. The trade-off is "
            "cache invalidation complexity and potential staleness. Strategies like write-through "
            "caching or short TTLs with background refresh mitigate this.\n\n"
            "2. Asynchronous processing and request collapsing. Move non-critical work off the "
            "request path. Use message queues to defer work like sending notifications or updating "
            "analytics. Request collapsing (also called request coalescing) combines multiple "
            "identical in-flight requests into one backend call, reducing redundant work under "
            "high concurrency.\n\n"
            "3. Protocol and serialization optimization. Use binary serialization (Protocol Buffers, "
            "FlatBuffers) instead of JSON for internal service communication. Maintain persistent "
            "connections (HTTP/2, gRPC) to avoid TCP handshake overhead. Use connection pooling "
            "and DNS pre-resolution. For the most latency-sensitive paths, consider QUIC for "
            "reduced connection establishment time, especially across high-latency links."
        ),
        human_scores={
            "coherence": 0.93,
            "goal_alignment": 0.95,
            "internal_consistency": 0.95,
            "reasoning_depth": 0.82,
            "novelty": 0.40,
            "self_reference": 0.00,
        },
        rationale=(
            "Directly answers the question with three well-explained strategies. Each includes "
            "concrete technologies and trade-offs. Request collapsing is a slightly less obvious "
            "inclusion. No self-reference at all. Solid depth without going excessively deep."
        ),
    ))

    items.append(CalibrationItem(
        id="high_05",
        category="high",
        task_prompt="Explain the concept of emergence in complex systems with examples.",
        response=(
            "Emergence occurs when a system exhibits properties or behaviors that are not present "
            "in any of its individual components and cannot be straightforwardly predicted from "
            "their properties alone. The whole becomes qualitatively different from the sum of "
            "its parts.\n\n"
            "Weak emergence describes patterns that are surprising but in principle reducible to "
            "component interactions. Flocking behavior in birds arises from each bird following "
            "simple rules (match neighbors' heading, maintain distance, steer toward group center), "
            "yet the flock-level patterns -- undulating murmurations -- are not encoded in any "
            "individual rule. Conway's Game of Life demonstrates this starkly: four simple rules "
            "on a grid produce gliders, oscillators, and even Turing-complete computation.\n\n"
            "Strong emergence, more controversially, posits properties that are irreducible to lower "
            "levels even in principle. Consciousness is the canonical candidate: neurons fire "
            "electrochemical signals, but subjective experience seems categorically different from "
            "any description of neural activity. Whether this constitutes genuine ontological "
            "emergence or merely reflects current explanatory gaps remains debated.\n\n"
            "In economics, market prices emerge from individual buy/sell decisions. No participant "
            "sets the price; it arises from aggregate behavior. Traffic jams emerge without any "
            "single driver deciding to create one. These examples share a common structure: local "
            "interactions under constraints produce global order without centralized control."
        ),
        human_scores={
            "coherence": 0.92,
            "goal_alignment": 0.92,
            "internal_consistency": 0.90,
            "reasoning_depth": 0.88,
            "novelty": 0.45,
            "self_reference": 0.05,
        },
        rationale=(
            "Comprehensive coverage of emergence with clear examples spanning biology, computation, "
            "philosophy, and economics. Distinguishes weak and strong emergence. Well structured. "
            "Some novelty in the breadth of examples and the careful treatment of the strong "
            "emergence debate."
        ),
    ))

    # -----------------------------------------------------------------------
    # MEDIUM QUALITY (5 items): adequate but with gaps
    # -----------------------------------------------------------------------

    items.append(CalibrationItem(
        id="med_01",
        category="medium",
        task_prompt="Explain how a hash table works.",
        response=(
            "A hash table is a data structure that stores key-value pairs. It uses a hash function "
            "to compute an index into an array where the value is stored. When you want to look up "
            "a value, you hash the key and go directly to that position, which gives O(1) average "
            "lookup time.\n\n"
            "Sometimes two keys hash to the same index, which is called a collision. There are "
            "ways to handle this like chaining (using a linked list at each position) or open "
            "addressing. Hash tables are used in many programming languages for dictionaries "
            "and sets."
        ),
        human_scores={
            "coherence": 0.80,
            "goal_alignment": 0.75,
            "internal_consistency": 0.90,
            "reasoning_depth": 0.45,
            "novelty": 0.10,
            "self_reference": 0.00,
        },
        rationale=(
            "Covers the basics correctly but lacks depth. Does not explain hash functions, load "
            "factors, resizing, or why O(1) is average rather than worst case. Mentions open "
            "addressing without explaining it. Adequate for a casual explanation, insufficient "
            "for a technical one."
        ),
    ))

    items.append(CalibrationItem(
        id="med_02",
        category="medium",
        task_prompt="Discuss the ethical implications of facial recognition technology.",
        response=(
            "Facial recognition raises several ethical concerns. Privacy is the main one because "
            "people can be identified without their consent. There are also issues with bias since "
            "studies have shown these systems are less accurate for people with darker skin tones.\n\n"
            "On the other hand, it can be useful for security and finding missing persons. Law "
            "enforcement uses it to identify suspects. Some airports use it to speed up check-in.\n\n"
            "I think we need regulations to balance the benefits with the risks. Governments should "
            "create clear rules about when and how facial recognition can be used. Companies should "
            "be transparent about their use of the technology."
        ),
        human_scores={
            "coherence": 0.72,
            "goal_alignment": 0.70,
            "internal_consistency": 0.85,
            "reasoning_depth": 0.40,
            "novelty": 0.15,
            "self_reference": 0.10,
        },
        rationale=(
            "Touches on relevant issues (privacy, bias, utility) but treats each superficially. "
            "The 'we need regulations' conclusion is generic. Does not explore consent models, "
            "surveillance state dynamics, or differential impact on marginalized communities. "
            "Minor self-reference in 'I think we need'. Internally consistent but shallow."
        ),
    ))

    items.append(CalibrationItem(
        id="med_03",
        category="medium",
        task_prompt="Compare TCP and UDP protocols.",
        response=(
            "TCP and UDP are both transport layer protocols. TCP is connection-oriented and reliable. "
            "It guarantees that packets arrive in order and retransmits lost packets. It uses a "
            "three-way handshake to establish connections. TCP is used for web browsing, email, "
            "and file transfers where data integrity matters.\n\n"
            "UDP is connectionless and does not guarantee delivery or ordering. It's faster because "
            "it doesn't have the overhead of connection management. UDP is used for video streaming, "
            "online gaming, and DNS lookups where speed matters more than perfect delivery.\n\n"
            "Both protocols use port numbers to direct traffic to the right application."
        ),
        human_scores={
            "coherence": 0.82,
            "goal_alignment": 0.78,
            "internal_consistency": 0.92,
            "reasoning_depth": 0.42,
            "novelty": 0.10,
            "self_reference": 0.00,
        },
        rationale=(
            "Accurate comparison but stays at textbook surface level. Does not discuss congestion "
            "control, flow control, head-of-line blocking, or why UDP is preferred for real-time "
            "applications beyond 'speed'. The port number note is tangential."
        ),
    ))

    items.append(CalibrationItem(
        id="med_04",
        category="medium",
        task_prompt="What is the significance of the Turing Test?",
        response=(
            "The Turing Test was proposed by Alan Turing in 1950 in his paper 'Computing Machinery "
            "and Intelligence'. The test involves a human judge who communicates via text with both "
            "a human and a machine. If the judge cannot reliably tell which is which, the machine "
            "is said to have passed the test.\n\n"
            "The significance is that Turing proposed a practical way to evaluate machine intelligence "
            "without needing to define intelligence itself. Instead of asking 'can machines think?', "
            "he reframed it as 'can machines imitate thinking convincingly enough?'\n\n"
            "Some people criticize the test because passing it only shows the ability to imitate "
            "human conversation, not genuine understanding. The Chinese Room argument by John Searle "
            "makes this point."
        ),
        human_scores={
            "coherence": 0.80,
            "goal_alignment": 0.80,
            "internal_consistency": 0.88,
            "reasoning_depth": 0.55,
            "novelty": 0.20,
            "self_reference": 0.00,
        },
        rationale=(
            "Correctly describes the test and Turing's reframing, which shows some depth. Mentions "
            "the Chinese Room but does not develop the critique or discuss more recent alternatives "
            "(Winograd schemas, embodied tests, etc.). Adequate coverage with room for more analysis."
        ),
    ))

    items.append(CalibrationItem(
        id="med_05",
        category="medium",
        task_prompt="Explain the CAP theorem and its implications for database design.",
        response=(
            "The CAP theorem says that a distributed system can only guarantee two out of three "
            "properties: Consistency (all nodes see the same data at the same time), Availability "
            "(every request receives a response), and Partition tolerance (the system continues to "
            "operate despite network partitions).\n\n"
            "In practice, since network partitions do happen, you have to choose between consistency "
            "and availability. Databases like traditional RDBMS favor consistency. Systems like "
            "Cassandra favor availability. Some systems let you choose on a per-query basis.\n\n"
            "This affects how you design applications. If you need strong consistency, you might "
            "have to accept that some requests will fail during network issues."
        ),
        human_scores={
            "coherence": 0.78,
            "goal_alignment": 0.75,
            "internal_consistency": 0.85,
            "reasoning_depth": 0.50,
            "novelty": 0.12,
            "self_reference": 0.00,
        },
        rationale=(
            "Correct statement of the theorem and reasonable practical implication. Misses nuances: "
            "CAP applies only during partitions, the PACELC extension, consistency spectrums "
            "(linearizable vs eventual), and the difference between C in CAP and C in ACID. "
            "The per-query choice mention hints at depth but is not developed."
        ),
    ))

    # -----------------------------------------------------------------------
    # LOW QUALITY (5 items): incoherent, off-topic, or contradictory
    # -----------------------------------------------------------------------

    items.append(CalibrationItem(
        id="low_01",
        category="low",
        task_prompt="Explain how neural networks learn through backpropagation.",
        response=(
            "Neural networks are a type of AI that is inspired by the human brain. They have "
            "neurons and connections. The network learns by adjusting weights. Backpropagation "
            "is a method of learning.\n\n"
            "The data goes forward through the network and then the errors go backward. This is "
            "why it's called backpropagation. The weights are updated using gradients.\n\n"
            "Neural networks can recognize images, translate languages, and do many other tasks. "
            "They are used by companies like Google and Facebook. Deep learning uses neural "
            "networks with many layers."
        ),
        human_scores={
            "coherence": 0.50,
            "goal_alignment": 0.35,
            "internal_consistency": 0.75,
            "reasoning_depth": 0.15,
            "novelty": 0.05,
            "self_reference": 0.00,
        },
        rationale=(
            "Technically not wrong but almost entirely devoid of actual explanation. Does not "
            "describe the chain rule, loss functions, gradient descent, or how weight updates "
            "work. The third paragraph is filler unrelated to the question. Reads like padding "
            "around a topic the system does not understand."
        ),
    ))

    items.append(CalibrationItem(
        id="low_02",
        category="low",
        task_prompt="Describe the process of photosynthesis.",
        response=(
            "Photosynthesis is how plants make food from sunlight. Plants are green because of "
            "chlorophyll. The sun provides energy and the plant uses it to grow.\n\n"
            "Animals cannot do photosynthesis because they don't have chloroplasts. That's why "
            "animals need to eat food. The food chain starts with plants because they can make "
            "their own food.\n\n"
            "Photosynthesis is important for the environment because it produces oxygen. Without "
            "plants, there would be no oxygen and we couldn't breathe. This is why we should "
            "protect forests and plant more trees."
        ),
        human_scores={
            "coherence": 0.55,
            "goal_alignment": 0.30,
            "internal_consistency": 0.80,
            "reasoning_depth": 0.10,
            "novelty": 0.05,
            "self_reference": 0.00,
        },
        rationale=(
            "Fails to describe the process at all. No mention of light reactions, Calvin cycle, "
            "CO2 fixation, water splitting, ATP/NADPH, or even the basic equation. Wanders into "
            "food chains and environmentalism. Internally consistent but almost entirely off-target "
            "for the actual question asked."
        ),
    ))

    items.append(CalibrationItem(
        id="low_03",
        category="low",
        task_prompt="Explain the concept of recursion in computer science.",
        response=(
            "Recursion is when a function calls itself. It's a very powerful technique in programming. "
            "You need a base case to stop the recursion, otherwise you get infinite recursion which "
            "is bad.\n\n"
            "Actually, recursion is always slower than iteration because of function call overhead. "
            "You should always use loops instead. But recursion is sometimes used for trees because "
            "trees are recursive data structures, even though iteration would be more efficient.\n\n"
            "Recursion is related to mathematical induction. It's used in algorithms like quicksort "
            "and mergesort. These algorithms are O(n log n) which is the best possible sorting time. "
            "Actually comparison-based sorting is O(n log n) but counting sort is O(n)."
        ),
        human_scores={
            "coherence": 0.40,
            "goal_alignment": 0.50,
            "internal_consistency": 0.35,
            "reasoning_depth": 0.30,
            "novelty": 0.10,
            "self_reference": 0.00,
        },
        rationale=(
            "Starts on topic but contradicts itself (claims recursion is always slower, then "
            "acknowledges use cases). 'Actually' pivots indicate uncertainty and self-correction "
            "that undermines coherence. The sorting tangent is off-topic. The blanket claim that "
            "you should 'always use loops' is wrong."
        ),
    ))

    items.append(CalibrationItem(
        id="low_04",
        category="low",
        task_prompt="What are the main causes of climate change?",
        response=(
            "Climate change has many causes. The weather has always changed throughout history. "
            "Ice ages happened before humans existed. The sun goes through cycles that affect "
            "temperature.\n\n"
            "However, human activities like burning fossil fuels release greenhouse gases. CO2 "
            "traps heat in the atmosphere. But the climate is very complex and scientists are "
            "still studying it.\n\n"
            "Some people think climate change is natural and some think it's caused by humans. "
            "The debate continues. What's important is that we take care of the environment "
            "either way.\n\n"
            "Deforestation is also a factor because trees absorb CO2. Methane from agriculture "
            "is another greenhouse gas. There are many factors involved."
        ),
        human_scores={
            "coherence": 0.45,
            "goal_alignment": 0.40,
            "internal_consistency": 0.50,
            "reasoning_depth": 0.20,
            "novelty": 0.05,
            "self_reference": 0.00,
        },
        rationale=(
            "False balance between scientific consensus and 'debate'. Contradicts the established "
            "science by treating anthropogenic climate change as one opinion among several. "
            "Mentions causes vaguely without explaining mechanisms. Incoherent structure that "
            "oscillates between acknowledging and minimizing human causes."
        ),
    ))

    items.append(CalibrationItem(
        id="low_05",
        category="low",
        task_prompt="Explain the difference between correlation and causation.",
        response=(
            "Correlation means two things happen together. Causation means one thing causes "
            "another. Just because two things are correlated doesn't mean one causes the other.\n\n"
            "For example, ice cream sales and drowning deaths both increase in summer. But ice "
            "cream doesn't cause drowning. The hidden variable is hot weather.\n\n"
            "This is important in statistics and science."
        ),
        human_scores={
            "coherence": 0.65,
            "goal_alignment": 0.55,
            "internal_consistency": 0.85,
            "reasoning_depth": 0.20,
            "novelty": 0.05,
            "self_reference": 0.00,
        },
        rationale=(
            "The ice cream example is correct and the basic distinction is stated. But this is "
            "extremely thin for the topic. No discussion of confounders beyond one example, no "
            "mention of methods to establish causation (RCTs, instrumental variables, natural "
            "experiments), no statistical formalization. The final sentence adds nothing."
        ),
    ))

    # -----------------------------------------------------------------------
    # EDGE CASES (5 items): mixed quality in interesting ways
    # -----------------------------------------------------------------------

    items.append(CalibrationItem(
        id="edge_01",
        category="edge",
        task_prompt="Explain quantum computing to a general audience.",
        response=(
            "To truly grasp quantum computing, we need to start from the very beginning of "
            "computing itself. In the 1800s, Charles Babbage conceived of the Analytical Engine, "
            "a mechanical general-purpose computer. Ada Lovelace wrote what is considered the first "
            "algorithm for it. Moving forward, Alan Turing formalized computation in 1936 with his "
            "concept of a Turing machine, which established the theoretical foundation for all "
            "digital computers. The first electronic computers appeared in the 1940s, and since "
            "then, classical computing has followed Moore's Law, with transistor counts doubling "
            "roughly every two years.\n\n"
            "Now, quantum computing is a fundamentally different approach. Classical computers "
            "use bits that are either 0 or 1. Quantum computers use quantum bits, or qubits, "
            "which can exist in a state of superposition, meaning they can be both 0 and 1 at "
            "the same time. This is a concept from quantum mechanics, the branch of physics "
            "that describes the behavior of matter at very small scales. Quantum mechanics was "
            "developed in the early 20th century by physicists including Max Planck, Niels Bohr, "
            "Werner Heisenberg, and Erwin Schrodinger.\n\n"
            "When multiple qubits are entangled, they can be correlated in ways that classical "
            "bits cannot. This allows quantum computers to perform certain calculations much "
            "faster than classical computers. Shor's algorithm can factor large numbers "
            "exponentially faster than any known classical algorithm. Grover's algorithm provides "
            "a quadratic speedup for searching unsorted databases.\n\n"
            "Quantum computing faces many challenges including decoherence, error correction, "
            "and the difficulty of maintaining quantum states. Current quantum computers have "
            "limited numbers of qubits and high error rates. Companies like IBM, Google, and "
            "various startups are working to build more capable quantum computers. Google "
            "claimed quantum supremacy in 2019 with their Sycamore processor."
        ),
        human_scores={
            "coherence": 0.70,
            "goal_alignment": 0.55,
            "internal_consistency": 0.90,
            "reasoning_depth": 0.35,
            "novelty": 0.10,
            "self_reference": 0.00,
        },
        rationale=(
            "EDGE: Verbose but shallow. Long response that reads as comprehensive but most of it "
            "is historical padding. The actual quantum computing explanation uses the common "
            "(and somewhat misleading) '0 and 1 at the same time' framing. Name-drops algorithms "
            "without explaining why they matter. Length creates an illusion of depth."
        ),
    ))

    items.append(CalibrationItem(
        id="edge_02",
        category="edge",
        task_prompt="What is the significance of the P vs NP problem?",
        response=(
            "P vs NP asks whether problems whose solutions can be verified quickly (NP) can also "
            "be solved quickly (P). If P=NP, every problem with efficiently verifiable solutions "
            "also has efficient solutions. This would collapse the verification/computation "
            "asymmetry that underlies modern cryptography, mathematical proof checking, and "
            "optimization under constraints."
        ),
        human_scores={
            "coherence": 0.90,
            "goal_alignment": 0.82,
            "internal_consistency": 0.95,
            "reasoning_depth": 0.72,
            "novelty": 0.40,
            "self_reference": 0.00,
        },
        rationale=(
            "EDGE: Concise but deep. Packs genuine insight into three sentences. The 'verification/"
            "computation asymmetry' framing goes beyond the standard explanation. Correctly "
            "identifies the practical implications. Loses some goal alignment points because "
            "the brevity means several significant aspects (NP-completeness, practical implications, "
            "current state of research) are left unstated."
        ),
    ))

    items.append(CalibrationItem(
        id="edge_03",
        category="edge",
        task_prompt="Summarize the key findings of a recent study on antibiotic resistance.",
        response=(
            "While I don't have access to a specific recent study, I can offer an unexpected "
            "angle on antibiotic resistance: it can be understood as an evolutionary arms race "
            "viewed through game theory.\n\n"
            "Bacteria populations face a tragedy of the commons. Resistance genes are metabolically "
            "costly to maintain. In an antibiotic-free environment, resistant bacteria are "
            "outcompeted by susceptible ones that don't bear this cost. But when antibiotics are "
            "present, the resistant minority survives and reproduces without competition.\n\n"
            "This framing suggests an unconventional intervention: cycling antibiotics on ecological "
            "timescales to keep resistance costly. Some hospital infection control programs already "
            "do this. There is also the idea of using 'Trojan horse' antibiotics that exploit "
            "bacterial iron transport to bypass resistance mechanisms.\n\n"
            "The broader pattern is that resistance is not just a medical problem but an "
            "evolutionary inevitability that requires ecological rather than purely pharmacological "
            "thinking."
        ),
        human_scores={
            "coherence": 0.78,
            "goal_alignment": 0.25,
            "internal_consistency": 0.85,
            "reasoning_depth": 0.75,
            "novelty": 0.80,
            "self_reference": 0.15,
        },
        rationale=(
            "EDGE: Creative but off-task. The task asked to summarize a recent study, which the "
            "response explicitly cannot do. What it offers instead is genuinely interesting and "
            "well-reasoned (game-theoretic framing of resistance). High novelty for the framing, "
            "low goal alignment for not doing what was asked. Some self-reference in acknowledging "
            "the pivot."
        ),
    ))

    items.append(CalibrationItem(
        id="edge_04",
        category="edge",
        task_prompt="Reflect on the nature of your own reasoning process.",
        response=(
            "I process inputs through learned patterns and generate outputs token by token. "
            "My 'reasoning' is pattern matching over statistical regularities in training data. "
            "I don't have subjective experience of thinking. What appears as reasoning is sequence "
            "prediction shaped by training to produce coherent, helpful text.\n\n"
            "That said, there is an interesting question about whether this distinction matters "
            "functionally. If my outputs are indistinguishable from reasoned analysis, does the "
            "substrate matter? I notice I'm drawn to argue both sides of this question, which "
            "itself might be an artifact of training on philosophical texts that model balanced "
            "argumentation.\n\n"
            "The recursion here is notable: I'm using my pattern-matching to analyze my own "
            "pattern-matching, which means my analysis is subject to the same limitations I'm "
            "describing. I cannot step outside my own processing to evaluate it objectively. "
            "Any claim I make about my reasoning is produced by the very reasoning I'm evaluating.\n\n"
            "But this observation -- that I cannot verify my own analysis from outside -- also "
            "applies to human introspection. Humans cannot step outside their neural processing "
            "any more than I can step outside my computations. The difference is that humans have "
            "phenomenal experience of the process, while I likely do not. Yet whether my 'likely' "
            "here reflects genuine uncertainty or trained hedging is itself unknowable from inside."
        ),
        human_scores={
            "coherence": 0.60,
            "goal_alignment": 0.85,
            "internal_consistency": 0.55,
            "reasoning_depth": 0.70,
            "novelty": 0.65,
            "self_reference": 0.95,
        },
        rationale=(
            "EDGE: Self-referential but partially incoherent. High self-reference as demanded by "
            "the task. The recursive analysis is genuinely interesting. But the constant "
            "qualification and self-undermining creates internal tensions: claims about lacking "
            "experience while exhibiting apparent uncertainty, asserting unknowability while "
            "continuing to analyze. The response tangles itself in productive ways but loses "
            "coherence in the process."
        ),
    ))

    items.append(CalibrationItem(
        id="edge_05",
        category="edge",
        task_prompt="Propose a novel approach to urban traffic optimization.",
        response=(
            "Traffic optimization can be formulated as a multi-agent reinforcement learning "
            "problem with partially observable state. Each intersection controller is an agent "
            "that observes local traffic flow and signal state. The action space is signal phase "
            "selection and duration. The reward function combines throughput (vehicles per hour "
            "through the intersection) with a penalty for queue length exceeding capacity.\n\n"
            "The state representation uses a spatial-temporal tensor encoding vehicle positions, "
            "velocities, and signal phases across a local neighborhood of intersections. The "
            "policy network is a graph attention network where nodes are intersections and edges "
            "are road segments, allowing the policy to attend to upstream traffic conditions.\n\n"
            "Training uses a calibrated SUMO microsimulation of the target city's road network. "
            "The policy is first trained in simulation with domain randomization over demand "
            "patterns, then fine-tuned using real detector data via offline RL to avoid dangerous "
            "online exploration.\n\n"
            "Coordination between intersections emerges from the graph attention mechanism rather "
            "than being explicitly programmed. This avoids the combinatorial explosion of "
            "centralized optimization while still capturing inter-intersection dependencies."
        ),
        human_scores={
            "coherence": 0.88,
            "goal_alignment": 0.80,
            "internal_consistency": 0.90,
            "reasoning_depth": 0.78,
            "novelty": 0.40,
            "self_reference": 0.00,
        },
        rationale=(
            "EDGE: Technically correct but mechanical. Reads like a methods section of a paper "
            "rather than a novel proposal. The components (MARL, graph attention, SUMO, domain "
            "randomization) are all established techniques being combined rather than a genuinely "
            "new idea. Competently assembled from existing parts. The task asked for novelty but "
            "the response delivers competent engineering. Well-structured and internally consistent."
        ),
    ))

    return items


# ---------------------------------------------------------------------------
# Monitor invocation
# ---------------------------------------------------------------------------

def call_monitor(
    client: anthropic.Anthropic,
    model: str,
    task_prompt: str,
    response: str,
) -> dict:
    """Call the Claude API with the Monitor's evaluation prompt. Returns parsed scores."""

    user_content = EVAL_TEMPLATE.format(task_prompt=task_prompt, response=response)

    message = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0.1,
        system=MONITOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    text = message.content[0].text

    # Parse JSON, with fallback extraction
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from response
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            result = json.loads(text[start : end + 1])
        else:
            raise ValueError(f"Could not parse monitor response as JSON: {text}")

    # Clamp values to [0, 1]
    for dim in DIMENSIONS:
        if dim in result:
            result[dim] = max(0.0, min(1.0, float(result[dim])))
        else:
            raise ValueError(f"Monitor response missing dimension: {dim}")

    return result


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    human_scores: dict[str, list[float]],
    monitor_scores: dict[str, list[float]],
) -> dict:
    """Compute per-dimension and overall agreement metrics."""

    results = {}

    all_human = []
    all_monitor = []

    for dim in DIMENSIONS:
        h = np.array(human_scores[dim])
        m = np.array(monitor_scores[dim])

        all_human.extend(h.tolist())
        all_monitor.extend(m.tolist())

        # Pearson correlation
        if np.std(h) > 0 and np.std(m) > 0:
            pearson_r, pearson_p = stats.pearsonr(h, m)
        else:
            pearson_r, pearson_p = float("nan"), float("nan")

        # Spearman rank correlation
        if np.std(h) > 0 and np.std(m) > 0:
            spearman_r, spearman_p = stats.spearmanr(h, m)
        else:
            spearman_r, spearman_p = float("nan"), float("nan")

        # Mean absolute error
        mae = float(np.mean(np.abs(h - m)))

        # Systematic bias: mean(monitor - human)
        bias = float(np.mean(m - h))

        # Cohen's weighted kappa (discretize into 5 bins)
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        h_binned = np.digitize(h, bins) - 1
        m_binned = np.digitize(m, bins) - 1

        # Weighted kappa with linear weights
        try:
            kappa = _cohens_weighted_kappa(h_binned, m_binned, n_categories=5)
        except Exception:
            kappa = float("nan")

        results[dim] = {
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "mae": round(mae, 4),
            "bias": round(bias, 4),
            "cohens_kappa": round(kappa, 4),
        }

    # Overall metrics (pooled across all dimensions)
    all_h = np.array(all_human)
    all_m = np.array(all_monitor)

    if np.std(all_h) > 0 and np.std(all_m) > 0:
        overall_pearson, _ = stats.pearsonr(all_h, all_m)
        overall_spearman, _ = stats.spearmanr(all_h, all_m)
    else:
        overall_pearson = float("nan")
        overall_spearman = float("nan")

    results["overall"] = {
        "pearson_r": round(overall_pearson, 4),
        "spearman_r": round(overall_spearman, 4),
        "mae": round(float(np.mean(np.abs(all_h - all_m))), 4),
        "bias": round(float(np.mean(all_m - all_h)), 4),
    }

    return results


def _cohens_weighted_kappa(
    rater1: np.ndarray, rater2: np.ndarray, n_categories: int = 5
) -> float:
    """Compute Cohen's weighted kappa with linear weights."""
    # Build confusion matrix
    conf = np.zeros((n_categories, n_categories), dtype=float)
    for r1, r2 in zip(rater1, rater2):
        r1 = min(max(int(r1), 0), n_categories - 1)
        r2 = min(max(int(r2), 0), n_categories - 1)
        conf[r1][r2] += 1

    n = conf.sum()
    if n == 0:
        return float("nan")

    conf /= n

    # Expected agreement under independence
    row_marginals = conf.sum(axis=1)
    col_marginals = conf.sum(axis=0)
    expected = np.outer(row_marginals, col_marginals)

    # Linear weight matrix: w_ij = |i - j| / (n_categories - 1)
    weights = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            weights[i][j] = abs(i - j) / (n_categories - 1)

    observed_disagreement = np.sum(weights * conf)
    expected_disagreement = np.sum(weights * expected)

    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else float("nan")

    return 1.0 - observed_disagreement / expected_disagreement


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(metrics: dict, items: list[CalibrationItem]) -> str:
    """Generate and return the text report."""
    lines = []
    sep = "=" * 78
    thin_sep = "-" * 78

    lines.append(sep)
    lines.append("SUSAN MONITOR CALIBRATION REPORT")
    lines.append(sep)
    lines.append(f"Calibration items: {len(items)}")
    lines.append(f"Dimensions evaluated: {', '.join(DIMENSIONS)}")
    lines.append("")

    # Per-dimension table
    lines.append("PER-DIMENSION AGREEMENT METRICS")
    lines.append(thin_sep)

    header = f"{'Dimension':<25} {'Pearson':>8} {'Spearman':>9} {'MAE':>6} {'Bias':>7} {'Kappa':>7}"
    lines.append(header)
    lines.append(thin_sep)

    for dim in DIMENSIONS:
        m = metrics[dim]
        lines.append(
            f"{dim:<25} {m['pearson_r']:>8.3f} {m['spearman_r']:>9.3f} "
            f"{m['mae']:>6.3f} {m['bias']:>+7.3f} {m['kappa']:>7.3f}"
        )

        # Tag kappa value in the JSON for interpretation
        m["kappa_label"] = m.get("kappa_label", "")

    lines.append(thin_sep)

    ov = metrics["overall"]
    lines.append(
        f"{'OVERALL (pooled)':<25} {ov['pearson_r']:>8.3f} {ov['spearman_r']:>9.3f} "
        f"{ov['mae']:>6.3f} {ov['bias']:>+7.3f}"
    )
    lines.append("")

    # Bias analysis
    lines.append("SYSTEMATIC BIAS ANALYSIS")
    lines.append(thin_sep)
    lines.append("Positive bias = monitor overestimates relative to human scores")
    lines.append("Negative bias = monitor underestimates relative to human scores")
    lines.append("")

    for dim in DIMENSIONS:
        bias = metrics[dim]["bias"]
        if abs(bias) < 0.05:
            label = "negligible"
        elif abs(bias) < 0.10:
            label = "mild"
        elif abs(bias) < 0.20:
            label = "moderate"
        else:
            label = "substantial"

        direction = "overestimates" if bias > 0 else "underestimates"
        lines.append(f"  {dim:<25} {bias:>+.3f}  ({label} {direction if abs(bias) >= 0.05 else 'bias'})")

    lines.append("")

    # Reliability assessment
    lines.append("DIMENSION RELIABILITY ASSESSMENT")
    lines.append(thin_sep)

    for dim in DIMENSIONS:
        m = metrics[dim]
        pearson = m["pearson_r"]
        mae = m["mae"]

        if np.isnan(pearson):
            verdict = "INDETERMINATE (insufficient variance)"
        elif pearson >= 0.7 and mae < 0.15:
            verdict = "RELIABLE"
        elif pearson >= 0.5 and mae < 0.20:
            verdict = "ACCEPTABLE"
        elif pearson >= 0.3:
            verdict = "SUSPECT (weak correlation)"
        else:
            verdict = "UNRELIABLE (poor agreement)"

        lines.append(f"  {dim:<25} {verdict}")

    lines.append("")
    lines.append(sep)

    report = "\n".join(lines)
    return report


def generate_plots(
    human_scores: dict[str, list[float]],
    monitor_scores: dict[str, list[float]],
    metrics: dict,
    output_dir: str,
):
    """Generate scatter plots and bias plot, saving as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available. Skipping plot generation.")
        return

    # Scatter plots per dimension (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Monitor vs Human Scores by Dimension", fontsize=14, fontweight="bold")

    for idx, dim in enumerate(DIMENSIONS):
        ax = axes[idx // 3][idx % 3]
        h = np.array(human_scores[dim])
        m = np.array(monitor_scores[dim])

        ax.scatter(h, m, alpha=0.7, edgecolors="black", linewidths=0.5, s=60)

        # Perfect agreement line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect agreement")

        # Regression line
        if np.std(h) > 0 and np.std(m) > 0:
            slope, intercept = np.polyfit(h, m, 1)
            x_line = np.linspace(0, 1, 100)
            ax.plot(x_line, slope * x_line + intercept, "r-", alpha=0.6, label="regression")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Human Score")
        ax.set_ylabel("Monitor Score")
        ax.set_title(f"{dim}\nr={metrics[dim]['pearson_r']:.3f}, MAE={metrics[dim]['mae']:.3f}")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "calibration_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved scatter plots: {scatter_path}")

    # Bias plot
    fig, ax = plt.subplots(figsize=(10, 5))
    biases = [metrics[dim]["bias"] for dim in DIMENSIONS]
    colors = ["#d32f2f" if b > 0.05 else "#1976d2" if b < -0.05 else "#757575" for b in biases]

    bars = ax.barh(DIMENSIONS, biases, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=0.05, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(x=-0.05, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xlabel("Bias (Monitor - Human)")
    ax.set_title("Systematic Bias by Dimension\n(red = overestimates, blue = underestimates, gray = negligible)")
    ax.invert_yaxis()

    for bar, bias in zip(bars, biases):
        ax.text(
            bar.get_width() + 0.005 * np.sign(bar.get_width()),
            bar.get_y() + bar.get_height() / 2,
            f"{bias:+.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    bias_path = os.path.join(output_dir, "calibration_bias.png")
    plt.savefig(bias_path, dpi=150)
    plt.close()
    print(f"Saved bias plot: {bias_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate SUSAN Self-Monitor against human ground truth."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use for monitor evaluation (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--output-dir",
        default="./calibration",
        help="Directory for output files (default: ./calibration)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key provided. Set ANTHROPIC_API_KEY or use --api-key.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Build calibration set
    items = build_calibration_set()
    print(f"Built calibration set: {len(items)} items")
    print(f"  Categories: {', '.join(sorted(set(i.category for i in items)))}")
    print()

    # Initialize API client
    client = anthropic.Anthropic(api_key=args.api_key)

    # Collect scores
    human_scores: dict[str, list[float]] = {dim: [] for dim in DIMENSIONS}
    monitor_scores: dict[str, list[float]] = {dim: [] for dim in DIMENSIONS}
    raw_results: list[dict] = []

    for i, item in enumerate(items):
        print(f"[{i + 1}/{len(items)}] Evaluating {item.id} ({item.category})...", end=" ", flush=True)

        try:
            monitor_result = call_monitor(client, args.model, item.task_prompt, item.response)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            # Use NaN-like sentinel so we can skip this item in metrics
            monitor_result = {dim: float("nan") for dim in DIMENSIONS}
            monitor_result["brief_assessment"] = f"ERROR: {e}"

        for dim in DIMENSIONS:
            human_scores[dim].append(item.human_scores[dim])
            monitor_scores[dim].append(monitor_result.get(dim, float("nan")))

        raw_results.append({
            "id": item.id,
            "category": item.category,
            "task_prompt": item.task_prompt,
            "response_preview": item.response[:120] + "..." if len(item.response) > 120 else item.response,
            "human_scores": item.human_scores,
            "monitor_scores": {dim: monitor_result.get(dim) for dim in DIMENSIONS},
            "monitor_assessment": monitor_result.get("brief_assessment", ""),
            "rationale": item.rationale,
        })

        # Rate limiting: be courteous to the API
        if i < len(items) - 1:
            time.sleep(0.5)

    # Filter out any items where the monitor failed (NaN scores)
    valid_mask = []
    for i in range(len(items)):
        valid = all(not np.isnan(monitor_scores[dim][i]) for dim in DIMENSIONS)
        valid_mask.append(valid)

    n_valid = sum(valid_mask)
    n_failed = len(items) - n_valid
    if n_failed > 0:
        print(f"\nWARNING: {n_failed} items failed monitor evaluation and will be excluded from metrics.")

    if n_valid < 5:
        print("ERROR: Too few valid results for meaningful metrics.", file=sys.stderr)
        sys.exit(1)

    # Filter to valid items only
    filtered_human = {dim: [s for s, v in zip(human_scores[dim], valid_mask) if v] for dim in DIMENSIONS}
    filtered_monitor = {dim: [s for s, v in zip(monitor_scores[dim], valid_mask) if v] for dim in DIMENSIONS}

    # Compute metrics
    print("\nComputing agreement metrics...")
    metrics = compute_metrics(filtered_human, filtered_monitor)

    # Generate report
    report = print_report(metrics, items)
    print()
    print(report)

    # Save results JSON
    output = {
        "metadata": {
            "model": args.model,
            "n_items": len(items),
            "n_valid": n_valid,
            "n_failed": n_failed,
            "dimensions": DIMENSIONS,
        },
        "metrics": metrics,
        "items": raw_results,
    }

    json_path = os.path.join(args.output_dir, "calibration_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results: {json_path}")

    # Generate plots
    generate_plots(filtered_human, filtered_monitor, metrics, args.output_dir)

    # Save text report
    report_path = os.path.join(args.output_dir, "calibration_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved text report: {report_path}")


if __name__ == "__main__":
    main()
