# Biology-Biochemistry-32

## Background

RNA molecules fold into secondary structures through complementary base pairing. These structures, in turn, strongly influence their biological functions. RNA inverse folding identifies nucleotide sequences that adopt a desired target structure as their unique optimal fold. This is particularly useful in synthetic biology, where functional RNAs are designed to form specific structures. The main challenge is that compatible nucleotides can form alternative pairings, making it difficult to guarantee a unique target fold. A tree-colouring method based on modulo-𝑚 separability uses structural colouring rules to prevent competing folds. Under a base-pair-maximization model, target structures in which every helix contains at least three consecutive base pairs can be designed in linear time.

## Problem

RNA inverse folding seeks a sequence for which a specified secondary structure is the unique optimum under a base-pair-maximization model. Solve the deterministic inverse-folding instance with target ((((((...)))(((...)))(((...)))))) and modulus 𝑚=2.
Fix the unpaired-leaf residue set as {0}. Process sibling pairs in increasing order of their left coordinates. During the depth-first colouring search, test colours in the order black, white, grey and return the first assignment satisfying local propriety and modulo-2 separation.
Determine whether the target satisfies the structural conditions, derive the deterministic colouring and corresponding RNA sequence, and independently verify uniqueness over all permissible noncrossing structures. Report the target pairs, helix information, forbidden motifs, colouring, designed sequence, maximum pairing score, number of optimal structures and uniqueness result. The final scalar must be the GC content of the designed sequence, expressed as a percentage and rounded to four decimal places.

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

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

step_1_parse_target

Goal
----
Parse a dot-bracket RNA target into ordered zero-based base-pair coordinates and unpaired coordinates.

```python
def parse_target(structure: str) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Return ordered base-pair coordinates and unpaired coordinates."""
    return None  # TODO: replace with the candidate implementation
```

### Step 2

step_2_analyse_target

Goal
----
Measure maximal helices and count the forbidden local motifs m5 and m3-bullet.

```python
def analyse_target(structure: str) -> tuple[int, int, int, int]:
    """Return four numeric structural-summary values."""
    return None  # TODO: replace with the candidate implementation
```

### Step 3

step_3_local_colouring

Goal
----
Determine whether a parent colour and its direct child colours satisfy the local proper-colouring limits.

```python
def check_local_coloring(parent_color: int, child_colors: tuple[int, ...]) -> int:
    """Return a numeric validity indicator."""
    return None  # TODO: replace with the candidate implementation
```

### Step 4

step_4_separated_colouring

Goal
----
Find the deterministic locally proper modulo-m separated colouring of the target.

```python
def find_separated_coloring(structure: str, m: int = 2) -> tuple[int, ...] | None:
    """Return deterministic colour codes aligned with the ordered base pairs."""
    return None  # TODO: replace with the candidate implementation
```

### Step 5

step_5_construct_sequence

Goal
----
Convert the ordered numeric colouring into a numerically encoded RNA sequence.

```python
def construct_sequence(structure: str, coloring: tuple[int, ...]) -> tuple[int, ...]:
    """Return the encoded RNA sequence induced by the ordered colour codes."""
    return None  # TODO: replace with the candidate implementation
```

### Step 6

step_6_evaluate_design

Goal
----
Verify compatibility and unique maximum-base-pair folding, then calculate GC percentage.

```python
def evaluate_design(
    sequence: tuple[int, ...],
    structure: str,
    minimum_span: int = 0,
    gc_decimals: int = 4,
) -> tuple[int, int, int, int, int, int, int, float]:
    """Return documented numeric verification values and rounded GC percentage."""
    return None  # TODO: replace with the candidate implementation
```

### Step 7

step_7_orchestrator

Goal
----
Orchestrate the six preceding RNA inverse-folding steps and return the final GC percentage.

```python
def run_pipeline(
    structure: str = "((((((...)))(((...)))(((...))))))",
    m: int = 2,
    minimum_span: int = 0,
    gc_decimals: int = 4,
) -> float:
    """Return the final numeric result from the complete public pipeline."""
    return None  # TODO: replace with the candidate implementation
```
