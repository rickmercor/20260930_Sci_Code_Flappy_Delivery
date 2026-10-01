# Biology-Ecology-29

## Background

The ecological niche of a species is often summarised by two numbers computed from a resource matrix, a
table of species by resource states (food types, habitats, host plants) holding the number of individuals
of each species recorded in each state: niche breadth, the extent to which a species spreads its use over
the states, and niche overlap, the extent to which two species use the same states. Because resource
states are rarely equivalent, a classical information-theoretic construction weights each state by how
much of the shared information between species identity and resource use that state carries, so that a
state used alike by every species counts for little and a state that separates the species counts for
much; the weights then enter adjusted resource-use probabilities, a standardized breadth that equals one for a species
using every state equally, and an overlap index that equals one for
species with identical adjusted distributions and zero for species using disjoint states. Weighting
factors derived from the shared information vanish for states that no species uses and for states used
in identical proportions by all species, which makes breadths undefined for some species, gives
counter-intuitive values when resource use is uniform, and makes the metrics unstable when the states are
subdivided into finer categories. Because a species' own data influence the weights, breadth and overlap
are evaluated noncircularly, with weights computed from the other species only. Refinements of the
weighting factors that keep every state positive and account for how much of the sampled resource space
is occupied restore well-defined, comparable metrics while preserving the classical construction, and
they can be evaluated exactly on small resource matrices.

## Problem

Niche breadth and niche overlap are the two metrics most used to describe how species share a set of
resource states (food types, microhabitats, host plants) recorded in a resource matrix of species by
states. The classical information-theoretic construction of these metrics weights every resource state
by how strongly its use profile distinguishes the species, but the classical relative weighting factor
vanishes for a state that no species uses and for a state used in identical proportions by all species,
which leaves metrics undefined for some matrices and makes values shift in magnitude and rank when the
resource states are subdivided. A recent source repairs this by replacing the relative weighting factor
with a modified factor that stays strictly positive for every state, depends on how much of the sampled
resource space is actually occupied, and keeps the rest of the classical construction (the adjusted
resource-use probabilities formed with a matrix-expansion constant, the standardized breadth and the
information-theoretic overlap, and the noncircular evaluation of both). The inputs are a resource matrix
and the constant; the outputs are per-species breadths and per-pair overlaps.

Take the resource matrix N (rows = six species, columns = seven resource states R1-R7, entries = numbers
of individuals recorded) below; R7 was sampled but holds no individuals of any species.

Sp1: 7  39  11   9   0   6  0
Sp2: 39  0  25   0  31   4  0
Sp3: 0   0   0  21  37  10  0
Sp4: 0   0   0   0  14   0  0
Sp5: 0   0   0  37   0   0  0
Sp6: 0  38   0   0   0   0  0

From N, form the pooled-community probabilities of the classical construction (the joint probability of
species and state, each species' own resource-use distribution, and the marginals of states and species)
and their entropies in natural logarithms (0 ln 0 = 0); from these, the contribution of each resource
state to the standardized resource heterogeneity, which the source turns into its modified weighting
factor. The number of occupied states that enters the modified factor is that of the complete matrix
(six of the seven states) throughout the analysis. With the constant k = 10,000, compute the source's adjusted
probabilities, its entropy-based (Shannon-type) standardized niche breadth of every species and its
information-theoretic overlap between species, all with the modified factor, and all noncircularly: the factors used for a
species' breadth are computed from the matrix without that species, and the factors used for a pair's
overlap from the matrix without both species of the pair. Identify the two species with the largest
noncircular breadths.

Report the noncircular niche overlap between those two species.

State the conventions you adopted and justify each from the source. In the reasoning give the
standardized resource heterogeneity of the complete matrix, the modified weighting factor of the empty
state R7 in the complete matrix, the identity and the noncircular breadths of the two broadest-niche
species, the largest of the seven modified factors used for that pair's overlap, the overlap of the same pair
obtained with factors from the complete matrix (as a contrast), and whether the reported overlap depends on
the value of k.

Output Format Requirements:

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.

You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.

Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

use_probabilities

Goal
----
Return the conditional resource-use probabilities for a resource matrix: resource_matrix[i, j] is the abundance (or resource-use value) of species i in resource state j, and the result holds, for every species, the probability that one of its individuals is associated with each resource state. A species whose row is entirely zero receives a row of zeros.

```python
def use_probabilities(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """Return the conditional resource-use probabilities for a resource matrix: resource_matrix[i, j] is the abundance (or resource-use value) of species i in resource state j, and the result holds, for every species, the probability that one of its individuals is associated with each resource state. A species whose row is entirely zero receives a row of zeros.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances: s species (rows) by r resource states (columns).

    Returns
    -------
    use_probabilities : numpy.ndarray
        Array of shape (s, r); each row sums to 1 (or is all zeros for an absent species) (float64).

    Raises
    ------
    ValueError
        If resource_matrix is not two-dimensional, contains a negative or non-finite entry, or has no positive entry.
    """
    return use_probabilities
```

### Step 2

resource_entropies

Goal
----
Return the four information quantities for a resource matrix, as the array [H(X), H(Y), H(XY), H_Y(X)]: the entropy of the resource states, the entropy of the species, their joint entropy, and the conditional entropy of the resource states given the species, all in nats, computed from the pooled-community probabilities.

```python
def resource_entropies(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """Return the four information quantities for a resource matrix, as the array [H(X), H(Y), H(XY), H_Y(X)]: the entropy of the resource states, the entropy of the species, their joint entropy, and the conditional entropy of the resource states given the species, all in nats, computed from the pooled-community probabilities.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances: s species by r resource states.

    Returns
    -------
    entropies : numpy.ndarray
        Array of shape (4,): [H(X), H(Y), H(XY), H_Y(X)] in nats (float64).

    Raises
    ------
    ValueError
        If resource_matrix is not two-dimensional, contains a negative or non-finite entry, or has no positive entry.
    """
    return entropies
```

### Step 3

state_contributions

Goal
----
Return the contribution of each resource state to the source's standardized resource heterogeneity M(X) (the quantity the source denotes delta_j), given the resource matrix, the use probabilities of step 01 and the entropies of step 02. The contributions sum to M(X); a state used by no species, or used in identical proportions by every species, contributes zero.

```python
def state_contributions(resource_matrix: "numpy.ndarray", use_probs: "numpy.ndarray",
                        entropies: "numpy.ndarray") -> "numpy.ndarray":
    """Return the contribution of each resource state to the source's standardized resource heterogeneity M(X) (the quantity the source denotes delta_j), given the resource matrix, the use probabilities of step 01 and the entropies of step 02. The contributions sum to M(X); a state used by no species, or used in identical proportions by every species, contributes zero.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances.
    use_probs : numpy.ndarray
        Array of shape (s, r) from step 01.
    entropies : numpy.ndarray
        Array of shape (4,) from step 02: [H(X), H(Y), H(XY), H_Y(X)].

    Returns
    -------
    state_contributions : numpy.ndarray
        Array of shape (r,): the contribution delta_j of every resource state (float64).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, an input is negative or non-finite, or H(X) is not positive (fewer than two occupied resource states).
    """
    return state_contributions
```

### Step 4

modified_weights

Goal
----
Return the source's modified weighting factor of each resource state (denoted ed_j) from the state contributions of step 03 and n_occupied, the number of resource states that are used by at least one species in the complete resource matrix the analysis refers to (the source's r'). The factors are strictly positive for every state, including states used by no species, and sum to 1.

```python
def modified_weights(contributions: "numpy.ndarray", n_occupied: int) -> "numpy.ndarray":
    """Return the source's modified weighting factor of each resource state (denoted ed_j) from the state contributions of step 03 and n_occupied, the number of resource states that are used by at least one species in the complete resource matrix the analysis refers to (the source's r'). The factors are strictly positive for every state, including states used by no species, and sum to 1. With delta_j the contribution of state j and r the number of states, ed_j = exp(delta_j n_occupied / r) / (sum over all states l of exp(delta_l n_occupied / r)).

    Parameters
    ----------
    contributions : numpy.ndarray
        Array of shape (r,) from step 03.
    n_occupied : int
        Number of resource states with a positive column total in the complete resource matrix, between 1 and r.

    Returns
    -------
    weights : numpy.ndarray
        Array of shape (r,) of strictly positive factors summing to 1 (float64).

    Raises
    ------
    ValueError
        If contributions is not a non-empty finite 1-D array, or n_occupied is not an integer between 1 and r.
    """
    return weights
```

### Step 5

adjusted_probabilities

Goal
----
Return the source's adjusted resource-use probabilities p*_ij of every species in a resource matrix for the given per-state weighting factors and the source's matrix-expansion constant k: each species' abundances are divided by its weighted, k-scaled total abundance Y*_i. A species whose row is entirely zero receives a row of zeros.

```python
def adjusted_probabilities(resource_matrix: "numpy.ndarray", weights: "numpy.ndarray",
                           k: float) -> "numpy.ndarray":
    """Return the source's adjusted resource-use probabilities p*_ij of every species in a resource matrix for the given per-state weighting factors and the source's matrix-expansion constant k: each species' abundances are divided by its weighted, k-scaled total abundance Y*_i. A species whose row is entirely zero receives a row of zeros.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances.
    weights : numpy.ndarray
        Array of shape (r,) of finite, nonnegative per-state weighting factors.
    k : float
        Positive matrix-expansion constant of the source (10000 in its analyses).

    Returns
    -------
    adjusted : numpy.ndarray
        Array of shape (s, r) of adjusted probabilities p*_ij (float64).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, an entry is negative or non-finite, k is not positive, or a species with positive abundance has zero weighted total.
    """
    return adjusted
```

### Step 6

niche_breadth

Goal
----
Return the source's standardized niche breadth beta' of one species from its row of adjusted probabilities (step 05), the per-state weighting factors used to form them, and the matrix-expansion constant k. The breadth equals 1 for a species that uses every resource state equally and is smaller for a species that concentrates its use on fewer states.

```python
def niche_breadth(adjusted_row: "numpy.ndarray", weights: "numpy.ndarray", k: float) -> float:
    """Return the source's standardized niche breadth beta' of one species from its row of adjusted probabilities (step 05), the per-state weighting factors used to form them, and the matrix-expansion constant k. The breadth equals 1 for a species that uses every resource state equally and is smaller for a species that concentrates its use on fewer states. With p*_j the species' adjusted probabilities and w_j the weighting factors, beta' = -(k / ln k) sum over j of w_j p*_j ln p*_j, with 0 ln 0 = 0.

    Parameters
    ----------
    adjusted_row : numpy.ndarray
        Array of shape (r,): the species' adjusted probabilities p*_ij.
    weights : numpy.ndarray
        Array of shape (r,) of finite, nonnegative per-state weighting factors.
    k : float
        Matrix-expansion constant of the source, greater than 1.

    Returns
    -------
    breadth : float
        The standardized niche breadth beta' of the species, as a native Python float.

    Raises
    ------
    ValueError
        If the arrays are not 1-D of the same length, an entry is negative or non-finite, k does not exceed 1, or the adjusted row has no positive entry.
    """
    return breadth
```

### Step 7

niche_overlap

Goal
----
Return the source's weighted information-theoretic niche overlap gamma' between two species from their rows of adjusted probabilities (step 05), the per-state weighting factors used to form them, and the matrix-expansion constant k. The overlap is 0 for species using disjoint resource states and 1 for species with identical adjusted distributions.

```python
def niche_overlap(adjusted_row_i: "numpy.ndarray", adjusted_row_h: "numpy.ndarray",
                  weights: "numpy.ndarray", k: float) -> float:
    """Return the source's weighted information-theoretic niche overlap gamma' between two species from their rows of adjusted probabilities (step 05), the per-state weighting factors used to form them, and the matrix-expansion constant k. The overlap is 0 for species using disjoint resource states and 1 for species with identical adjusted distributions. With a_j and b_j the two species' adjusted probabilities, w_j the weighting factors and I(x) = x ln x (0 ln 0 = 0), gamma' = -(1 / (2 ln 2)) sum over j of w_j k [I(a_j) + I(b_j) - I(a_j + b_j)].

    Parameters
    ----------
    adjusted_row_i : numpy.ndarray
        Array of shape (r,): adjusted probabilities of the first species.
    adjusted_row_h : numpy.ndarray
        Array of shape (r,): adjusted probabilities of the second species.
    weights : numpy.ndarray
        Array of shape (r,) of finite, nonnegative per-state weighting factors.
    k : float
        Positive matrix-expansion constant of the source.

    Returns
    -------
    overlap : float
        The niche overlap gamma' between the two species, as a native Python float.

    Raises
    ------
    ValueError
        If the arrays are not 1-D of the same length, an entry is negative or non-finite, k is not positive, or either adjusted row has no positive entry.
    """
    return overlap
```

### Step 8

generalist_pair_overlap

Goal
----
Orchestrator. For a resource matrix and the matrix-expansion constant k, return the source's noncircular niche overlap gamma' (step 07) between the two species with the largest noncircular niche breadths beta' (step 06), every weighting factor being the source's modified factor of step 04 built from steps 01-03 under the source's noncircularity rule: the factors used for one species' breadth come from the resource matrix with that species removed, and the factors used for a pair's overlap come from the matrix with both species removed, while the count of occupied states passed to step 04 is always that of the complete matrix. Adjusted probabilities (step 05) of the focal species are formed from their own rows with those factors. Raise ValueError if the two largest breadths are not uniquely determined (a tie at the first or second rank) or if fewer than three species are present. Call the earlier step functions rather than reimplementing them.

```python
def generalist_pair_overlap(resource_matrix: "numpy.ndarray", k: float) -> float:
    """Orchestrator. For a resource matrix and the matrix-expansion constant k, return the source's noncircular niche overlap gamma' (step 07) between the two species with the largest noncircular niche breadths beta' (step 06), every weighting factor being the source's modified factor of step 04 built from steps 01-03 under the source's noncircularity rule: the factors used for one species' breadth come from the resource matrix with that species removed, and the factors used for a pair's overlap come from the matrix with both species removed, while the count of occupied states passed to step 04 is always that of the complete matrix. Adjusted probabilities (step 05) of the focal species are formed from their own rows with those factors. Raise ValueError if the two largest breadths are not uniquely determined (a tie at the first or second rank) or if fewer than three species are present. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r), s >= 3, of finite, nonnegative abundances.
    k : float
        Matrix-expansion constant of the source, greater than 1.

    Returns
    -------
    overlap : float
        The noncircular niche overlap gamma' of the two broadest-niche species, as a native Python float.

    Raises
    ------
    ValueError
        If resource_matrix is invalid for the earlier steps, has fewer than three species, k does not exceed 1, or the two largest noncircular breadths are tied.
    """
    return overlap
```
