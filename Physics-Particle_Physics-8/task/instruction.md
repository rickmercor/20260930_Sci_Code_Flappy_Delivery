# Phase-sector diversity for colour-ordered antenna mappings

## Background

# Scientific background

Infrared subtraction turns divergent real-emission contributions into objects that can be
integrated numerically.  A local counterterm must reproduce the real matrix element in every soft
and collinear limit, while its mapped hard momenta must approach the correct reduced event.  An
antenna mapping absorbs unresolved recoil onto two hard radiators, but its input is a colour chain
rather than an unordered set.  Non-adjacent particles in that chain cannot simply become one
collinear cluster.

With several unresolved emissions, more than one chain can be physically relevant.  The source
paper resolves this by assigning candidate ordered maps through deterministic sectors.  Its three
triple-unresolved constructions use different primary invariants, compound invariants, nested
branches, comparison directions, case numbering, and associations between a listed case and a
colour chain.  Those source-specific choices are the algorithm to be recovered; superficially
similar permutations are not interchangeable.

The numerical ensemble uses RAMBO, a uniform generator for massless phase space.  Four uniform
values generate each unconstrained lightlike vector.  A common Lorentz boost then takes the event
to its centre-of-mass frame, and a common scale fixes total energy to one.  Every output particle
remains future-directed and massless.  Lorentz transformations and common rescalings leave the
sector decisions unchanged because their predicates are homogeneous in Mandelstam invariants.

Sector populations show whether the branches of each taxonomy are reached.  Mapping-order
disagreement ignores the taxonomy-local sector number and directly compares the five particle
labels selected on the same event.  The final index adds mean pairwise disagreement to mean
normalized sector entropy.  The source paper supplies the sector algorithms, while the seeded
ensemble and scalar diagnostic make their comparison reproducible.

## Problem

Higher-order QCD subtraction calculations repeatedly map a many-parton configuration onto two
hard radiator momenta.  An antenna map is colour ordered: it approaches the correct hard
configuration only when the unresolved momenta are supplied in a chain compatible with the
infrared limit.  The source paper defines deterministic phase-space sectors for choosing
that chain in triple-unresolved configurations.

Reproduce the construction for all three antenna classes in Sections 3.2.1--3.2.3.  Use the
sector labels and particle orders in the paper's listing; those case identifiers select the
downstream mapping.

Use seven functions:

    def assign_unordered_antenna_sectors(momenta: np.ndarray) -> np.ndarray
    def assign_ordered_pair_antenna_sectors(momenta: np.ndarray) -> np.ndarray
    def assign_multidipole_antenna_sectors(momenta: np.ndarray) -> np.ndarray
    def generate_five_parton_events(seed: int, n_events: int) -> np.ndarray
    def tabulate_sector_populations(unordered: np.ndarray, ordered_pair: np.ndarray,
                                    multidipole: np.ndarray) -> np.ndarray
    def compute_ordering_disagreements(unordered: np.ndarray, ordered_pair: np.ndarray,
                                       multidipole: np.ndarray) -> np.ndarray
    def compute_sector_mapping_index(seed: int, n_events: int) -> float

The first three functions return an integer array with columns
`(sector, label_1, label_2, label_3, label_4, label_5)`.  Sector numbers are 1-based positions
in the paper's own listing.  Particle labels are also 1-based.  Particles 1 and 5 are the hard
radiators, so every mapping order begins with 1 and ends with 5.  Momenta use row order
`[E, px, py, pz]`, signature `(+,-,-,-)`, and `s_ij = 2 p_i dot p_j`.  An index that stands for
a cluster carries that cluster's total momentum under the same rule, so
`s_i(j+k) = 2 p_i dot (p_j + p_k)`.  Every function raises `ValueError` for
input outside its stated domain.  The momenta these three functions are given are
future-directed, massless, and free of exact comparison ties.

The evaluated ensemble is deterministic RAMBO phase space with total four-momentum
`(1,0,0,0)`.  Instantiate `rng = numpy.random.default_rng(int(seed))` and make exactly one
draw `u = rng.random((int(n_events), 5, 4))`.  Consume that array in its indexed order: event,
then particle labels 1 through 5, then the four values for that particle.  Use columns 0, 1, 2,
and 3 respectively in `c = 2*u0-1`, `phi = 2*pi*u1`, and `E = -log(u2*u3)`.  Do not replace
this shaped draw with four component-wise draws or a transposed draw layout.

The final index is the mean of the three pairwise mapping-order disagreement fractions plus the
mean normalized Shannon entropy of the three sector distributions, with sector counts
`(12,12,5)`.  It lies in `[0,2]` and is large only when the antenna classes both select different
colour chains and broadly occupy their available sectors.

Run the complete pipeline with:

    compute_sector_mapping_index(seed=20260829, n_events=32768)

Round the returned value to ten digits after the decimal point.

In `<reasoning>`, give a compact verification certificate that:

- identifies the candidate and compound invariants, comparison directions, paper-listing sector
  convention, and case-to-chain associations that distinguish the three taxonomies, enumerating
  for every listed case its comparison, its sector number and its five-particle chain (a compact
  table is fine);
- explains what the sector construction buys the calculation and states the structural property
  the source establishes for the listed cases of each taxonomy;
- reports the dominant sector and its population for each taxonomy, the three disagreement
  fractions, the three normalized entropies in the same taxonomy sequence used by the function
  arguments above, their two means, and the unrounded index; and
- reports, as the RNG fingerprint, the energy of the first particle of the first event of the
  returned batch, together with quantitative closure figures for the batch's total four-momentum,
  reported separately for the time component and for the largest space component, and for how far
  each particle sits off its light cone.

Report every quantity listed above, including the diagnostic ones that do not enter the index;
a compact table is permitted for these diagnostics.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Include the requested diagnostic scalars and only
the few additional scalars that determine the final number.
Do not paste event arrays, complete population rows, or per-event sector tables.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Include the requested diagnostic scalars and only
the few additional scalars that determine the final number.
Do not paste event arrays, complete population rows, or per-event sector tables.
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

01_assign_unordered_antenna_sectors

Goal
----
Assign the paper's sectors for three unordered emissions.

```python
import numpy as np

def assign_unordered_antenna_sectors(momenta: np.ndarray) -> np.ndarray:
    '''Return the Section 3.2.1 sector and mapping order.

    Returns
    -------
    assignment : np.ndarray, shape (N,6), int64
        Sector, then the five 1-based particle labels in mapping order.'''
    return np.empty((0, 6), dtype=np.int64)
```

### Step 2

02_assign_ordered_pair_antenna_sectors

Goal
----
Assign sectors for one unordered emission and an ordered pair.

```python
import numpy as np

def assign_ordered_pair_antenna_sectors(momenta: np.ndarray) -> np.ndarray:
    '''Return the Section 3.2.2 sector and mapping order.

    Returns
    -------
    assignment : np.ndarray, shape (N,6), int64
        Sector, then the five 1-based particle labels in mapping order.'''
    return np.empty((0, 6), dtype=np.int64)
```

### Step 3

03_assign_multidipole_antenna_sectors

Goal
----
Assign sectors for a gluon emitted between multiple dipoles.

```python
import numpy as np

def assign_multidipole_antenna_sectors(momenta: np.ndarray) -> np.ndarray:
    '''Return the Section 3.2.3 sector and mapping order.

    Returns
    -------
    assignment : np.ndarray, shape (N,6), int64
        Sector, then the five 1-based particle labels in mapping order.'''
    return np.empty((0, 6), dtype=np.int64)
```

### Step 4

04_generate_five_parton_events

Goal
----
Generate deterministic five-particle massless phase space.

```python
import numpy as np

def generate_five_parton_events(seed: int, n_events: int) -> np.ndarray:
    '''Return a deterministic RAMBO batch.

    Returns
    -------
    events : np.ndarray, shape (n_events,5,4), float
        Massless momenta in [E,px,py,pz] order and unit COM energy.'''
    return np.empty((0, 5, 4), dtype=float)
```

### Step 5

05_tabulate_sector_populations

Goal
----
Tabulate sector occupancies for all three taxonomies.

```python
import numpy as np

def tabulate_sector_populations(unordered: np.ndarray, ordered_pair: np.ndarray, multidipole: np.ndarray) -> np.ndarray:
    '''Return zero-padded sector-frequency rows.

    Returns
    -------
    populations : np.ndarray, shape (3,12), float
        Per-taxonomy sector fractions in fixed row and sector order.'''
    return np.empty((3, 12), dtype=float)
```

### Step 6

06_compute_ordering_disagreements

Goal
----
Measure pairwise disagreement of the selected colour chains.

```python
import numpy as np

def compute_ordering_disagreements(unordered: np.ndarray, ordered_pair: np.ndarray, multidipole: np.ndarray) -> np.ndarray:
    '''Return pairwise mapping-order disagreement fractions.

    Returns
    -------
    disagreements : np.ndarray, shape (3,), float
        Fractions in (unordered/ordered-pair, ordered-pair/multidipole, unordered/multidipole) order.'''
    return np.zeros(3, dtype=float)
```

### Step 7

07_compute_sector_mapping_index

Goal
----
Compose the phase-sector taxonomy diversity index.

```python
import numpy as np

def compute_sector_mapping_index(seed: int, n_events: int) -> float:
    '''Return the unrounded mapping-taxonomy diversity index.

    Returns
    -------
    mapping_index : float
        Mean ordering disagreement plus mean normalized sector entropy.'''
    return 0.0
```
