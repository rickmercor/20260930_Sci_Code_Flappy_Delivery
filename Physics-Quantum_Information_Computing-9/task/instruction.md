# Physics-Quantum_Information_Computing-9

## Background

The source paper augments a distributed k-hop entanglement-routing pipeline with fidelity-aware Phase-4 recovery. Base Q-GUARD assigns a recovery segment a hop-count share of the end-to-end Werner budget and uses equal per-hop targets on each detour. Q-GUARD-WS instead uses calibrated link qualities to assign the replaced segment a depolarization-burden share, predicts link fidelities, and greedily removes unnecessary purification rounds while retaining the segment budget.

Recovery spans are ranked by the Phase-4 expected-goodput metric, which combines reserved width, all-intermediate-swap success, analytic purification cost, and the supplied bottleneck availability. This is distinct from Q-GUARD-FP's exploratory Phase-2 purification-aware path-search score. The benchmark deliberately tests the main recovery mechanism and the weighted-split extension, not Q-GUARD-FP.

The paper's planning calculation uses its ideal symmetric BBPSSW fidelity recurrence and analytic purification-resource convention. Runtime purification is stochastic and can use asymmetric realized pairs, but those runtime effects are outside this deterministic planning benchmark.

## Problem

Implement a deterministic comparison of base Q-GUARD and Q-GUARD-WS recovery-span planning by reconstructing the method from the cited primary literature; this new instance is not a worked example from the paper.

A seven-hop major path has request threshold 0.80, calibrated link qualities [5.2, 9.0, 4.8, 11.5, 6.0, 10.0, 7.5] in source-to-destination order, and replaced zero-based links 2 and 3; intermediate swaps succeed independently with probability 0.91 and each link permits at most four ideal symmetric purification rounds.

Compare the following detours in listed zero-based order, where `fidelities` are realized pre-purification values for base Q-GUARD, `qualities` are calibrated hardware parameters for Q-GUARD-WS, `width` is reserved span width, and `availability` is the precomputed bottleneck availability factor.

[
    {"width":20, "availability":0.623,
     "fidelities":[0.963,0.927,0.911,0.900],
     "qualities":[14.56,12.46,6.22,8.94]},
    {"width":28, "availability":0.738,
     "fidelities":[0.933,0.979,0.969,0.962],
     "qualities":[9.47,16.68,14.42,14.78]},
    {"width":20, "availability":0.784,
     "fidelities":[0.951,0.926],
     "qualities":[14.68,12.63]},
    {"width":28, "availability":0.859,
     "fidelities":[0.951,0.912],
     "qualities":[10.51,7.96]},
]

Complete the eight ordered functions with Python's standard library, applying each paper-defined recovery allocation to its corresponding inputs; a span is eligible only if its plan obeys the round and width limits and its recomposed Werner product meets the allocated segment budget within 1e-12.

Rank eligible spans by the paper's Phase-4 EXG score with earlier input order breaking exact ties; each method has at least two eligible candidates.

The final scalar is Q-GUARD-WS's selected EXG minus base Q-GUARD's selected EXG, with full precision retained through selection and subtraction and Python `round(value, 6)` applied exactly once; in `<reasoning>`, report the two segment budgets and, for each method, the selected index, selected round vector, Phase-4 purification denominator, and selected EXG.

Also report a local robustness certificate using the same paper-defined planner and ranking rule: first, lower the request fidelity continuously from 0.80 while tracking the baseline Q-GUARD-WS winner, reporting the first request-fidelity value at which its round plan changes and the plan immediately below that transition. Second, reset the baseline and lower only the second `qualities` entry of candidate 2 continuously from 12.63, replanning and reranking after each change: report the first quality at which candidate 2's round plan changes and its plan immediately below, then the first quality at which the selected Q-GUARD-WS candidate changes and the new selected index immediately below. All other inputs remain fixed, equality is evaluated with the stated 1e-12 feasibility convention, and each transition is approached from the original baseline side.

Report budgets and all three transition values to nine decimal places, EXG scores to six decimal places, the unrounded selected-score difference to twelve decimal places, and integer certificates exactly; do not print BBPSSW trajectories or full candidate tables.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

werner_parameter

Goal
----
Convert one Werner-state fidelity to the parameter that composes multiplicatively under swapping.

```python
import math

def werner_parameter(fidelity):
    """Return the Werner parameter of one Bell-pair fidelity.

    Returns:
        float: A finite scalar.
    """
    return float(w)
```

### Step 2

hardware_initial_fidelity

Goal
----
Predict a freshly generated pair's initial fidelity from a calibrated link-quality parameter.

```python
import math

def hardware_initial_fidelity(quality):
    """Return the paper's noise-free predicted initial fidelity.

    Returns:
        float: A fidelity in (0.25, 1).
    """
    return float(fidelity)
```

### Step 3

bbpssw_after_rounds

Goal
----
Propagate a Werner-state fidelity through a specified number of ideal symmetric BBPSSW rounds.

```python
import math

def bbpssw_after_rounds(initial_fidelity, rounds):
    """Return the fidelity after exactly the requested BBPSSW rounds.

    Returns:
        float: The full-precision post-round fidelity.
    """
    return float(fidelity)
```

### Step 4

segment_budgets

Goal
----
Compute the base and hardware-weighted Werner budgets assigned to a replaced major-path segment.

```python
import math

def segment_budgets(fidelity_threshold, major_qualities, replaced_indices):
    """Return the two segment budgets and their burden diagnostics.

    Returns:
        list[float]: Five values in the documented order.
    """
    return [threshold_w, base_budget, weighted_budget, segment_burden, total_burden]
```

### Step 5

equal_split_plan

Goal
----
Plan the minimum per-hop purification rounds for base Q-GUARD's equal target allocation.

```python
import math

def equal_split_plan(initial_fidelities, segment_budget, r_max, width):
    """Return base Q-GUARD's ordered per-hop round plan.

    Returns:
        list[int]: One round count per input hop, with -1 marking an unreachable hop.
    """
    return rounds
```

### Step 6

weighted_split_plan

Goal
----
Allocate Q-GUARD-WS purification rounds greedily across heterogeneous detour links.

```python
import math

def weighted_split_plan(qualities, segment_budget, r_max, width):
    """Return Q-GUARD-WS's ordered nonuniform round plan.

    Returns:
        list[int]: One count per quality, or all -1 when infeasible.
    """
    return rounds
```

### Step 7

recovery_exg

Goal
----
Evaluate the paper's local expected-goodput score for one feasible recovery span.

```python
import math

def recovery_exg(width, swap_probability, rounds, availability_factor):
    """Return the expected goodput of one feasible span.

    Returns:
        float: The full-precision EXG score.
    """
    return float(exg)
```

### Step 8

qguard_ws_benchmark

Goal
----
Run the complete recovery comparison and return the selected-EXG gain of Q-GUARD-WS over base Q-GUARD.

```python
import math

def qguard_ws_benchmark(fidelity_threshold=0.80, major_qualities=None, replaced_indices=None, swap_probability=0.91, r_max=4, candidates=None):
    """Return the six-decimal selected-EXG gain for one recovery instance; omitted arguments use the prompt benchmark.

    Returns:
        float: Q-GUARD-WS selected EXG minus base selected EXG, rounded once.
    """
    return float(result)
```
