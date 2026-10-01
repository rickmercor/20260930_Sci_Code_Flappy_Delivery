# Physics-Particle_Physics-34

## Background

In a ribbon-graph organization of cubic scalar amplitudes, a bordered surface specifies a family of diagrams rather than one diagram. Cutting an admissible curve exposes a propagator, and a triangulation is reached by a sequence of cuts. This importance measure is defined across the diagram family, so the number of different topological endpoints alone does not specify the sampling distribution.

The requested observable is a dimensionless collision probability for topology after one cut. Each component retains the association between its genus and its boundary mark counts when labels are forgotten. The genus and marked boundaries are the full finite input, with masses and kinematics fixed in the problem; neither experimental data nor sampled random numbers enter this computation.

## Problem

Consider the recursive first-propagator distribution used to importance-sample the all-diagram contribution of massive cubic matrix-scalar theory in two spacetime dimensions, with zero external momenta and unit internal masses. The diagrams are encoded by a connected orientable unpunctured surface of genus one with two boundary components carrying two and three cyclically ordered external marks; compute the probability that two independent first cuts produce the same unlabelled cut surface under the published surface-level measure obtained by replacing positive polynomial sums by their dominant monomials.

The curve sum is modulo orientation-preserving mapping classes that fix the labeled external marks, so distinct cuts with the same topology retain their multiplicities. For the reported event only, forget external labels and boundary names after the first cut: describe each connected component by its genus paired with its sorted boundary mark counts, and declare outcomes identical when the unordered multisets of these paired descriptors coincide, including the new endpoint marks created by the cut.

Use the surface recursion associated with this sampler, including its disconnected-surface normalization and empty-triangulation convention. Report a single number, determined exactly or to absolute numerical error below 1e-12 before rounding it to ten digits after the decimal point; no random realization is requested.

In concise reasoning identify the source-defined weighting rule and give the total pre-normalization first-cut weight, the number of aggregated outcomes and the sum of squared aggregated unnormalized weights. These refer to the stated unlabelled event, not equality of labeled curves or completed triangulations.

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

01_surface_descriptor

Goal
----
A connected orientable unpunctured surface is specified by genus g and positive boundary mark counts. For cubic scalar ribbon graphs, its triangulation has E=sum(marks)+3*b+6*g-6 arcs and L=2*g+b-1 loops, where b counts boundaries. At spacetime dimension 2 its degree is d=E-L. Return (g, sorted_marks, E, L, d), preserving repeated boundary counts. The finite domain has g in {0,1}, one through three boundaries and E from 0 through 11. The only allowed E=0 surface is the triangle (0,(3,)); disks with one or two marks are unstable. Accept integer scalars but not booleans or floating-point values, and raise ValueError outside the domain.

```python
from numbers import Integral


def surface_descriptor(genus, marks):
    """Return the canonical numeric surface descriptor for the stated finite domain."""
    return None
```

### Step 2

02_nonseparating_cuts

Goal
----
Enumerate nonseparating cuts on a connected unpunctured surface. Accept genus 0 or 1, one through three positive integer boundary counts and E=sum(marks)+3*b+6*g-6 from 0 through 11, with only the triangle allowed at E=0. Reject booleans, floating-point counts and inputs outside this domain with ValueError. Return raw records (components, multiplicity), where components is a tuple containing one (genus, sorted_marks) descriptor. Do not aggregate records.

```python
def nonseparating_cuts(genus, marks):
    """Return raw numeric records for boundary merging and genus lowering."""
    return None
```

### Step 3

03_separating_cuts

Goal
----
Enumerate raw separating cuts on a connected orientable unpunctured surface. Accept genus 0 or 1, one through three positive integer boundary counts and E=sum(marks)+3*b+6*g-6 from 0 through 11, with only the triangle allowed at E=0. Reject booleans, floating-point counts and out-of-domain inputs with ValueError. Sort parent marks, retaining repeated entries as distinct indexed boundaries.

```python
def separating_cuts(genus, marks):
    """Return raw numeric records for the separating-curve family."""
    return None
```

### Step 4

04_aggregate_cuts

Goal
----
Combine the raw nonseparating and separating curve families into unlabelled cut outcomes. Accept genus 0 or 1, one through three positive integer boundary counts and E=sum(marks)+3*b+6*g-6 from 0 through 11, with only the triangle allowed at E=0. Invalid counts, booleans and floating-point counts raise ValueError. Each output record is (components,multiplicity), sorted lexicographically by components. Each component is (genus,sorted_marks), and the components form an unordered multiset represented by a sorted tuple.

```python
def aggregate_cuts(genus, marks):
    """Return sorted unlabelled cut records with summed multiplicities."""
    return None
```

### Step 5

05_reduced_surface_weight

Goal
----
Evaluate the reduced tropical weight h for a connected surface in spacetime dimension 2. Accept genus 0 or 1, one through three positive integer boundary counts and E=sum(marks)+3*b+6*g-6 from 0 through 11, with only the triangle allowed at E=0. Invalid counts, booleans and floating-point counts raise ValueError. The loop count is L=2*g+b-1 and the degree is d=E-L. The triangle (0,(3,)) has h=1 even though d=0.

```python
from fractions import Fraction
from functools import lru_cache


def reduced_surface_weight(genus, marks):
    """Return the connected reduced weight, including the triangle convention."""
    return None
```

### Step 6

06_first_cut_distribution

Goal
----
Return (total_weight, probabilities) for the aggregated first-cut topology distribution, with probabilities in lexicographic component-descriptor order. Accept genus 0 or 1, one through three positive integer boundary mark counts and E=sum(marks)+3*b+6*g-6 from 1 through 11. Booleans, floating-point counts and other inputs raise ValueError; the E=0 triangle is not a first-cut query.

```python
from math import fsum


def first_cut_distribution(genus, marks):
    """Return the total first-cut weight and its normalized probability tuple."""
    return None
```

### Step 7

07_surface_collision

Goal
----
Compose the canonical surface descriptor, nonseparating and separating cut enumeration, multiplicity aggregation, reduced surface recursion and first-cut normalization to return the probability that two independent first-cut draws have the same unlabelled component collection. Accept genus 0 or 1, one through three positive integer boundary counts and E=sum(marks)+3*b+6*g-6 from 1 through 11. Reject booleans, floating-point counts and other inputs with ValueError. The terminal triangle has no first-cut distribution and is rejected here.

```python
from math import fsum


def surface_collision(genus, marks):
    """Return the aggregated first-cut topology collision probability."""
    return None
```
