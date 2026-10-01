# Biology-Ecology-19

## Background

Stability in the usual sense asks whether a steady state is attracting. That question is silent about the first instant after a disturbance. A community can be attracting and still grow a perturbation at $t=0$.

When that early growth is concentrated on a few feeding links, those links are the practical place to look. The community matrix used here is written at an unknown steady state from topology, biomass-flow shares, and elasticities. It is not taken from one named ODE.

## Problem

A food web can pull nearby states back as time runs forward and still amplify a shock at the first instant. The early jump is worth locating on a small feeding pattern only when the leading short-time mode of the community matrix already lives on those nodes. Ranking every subgraph by a bound is a different test, and the two tests need not name the same pattern.
Species are numbered 1 to 6. Entry $A_{ij}$ is 1 when $i$ eats $j$:
$A=\begin{pmatrix}0&0&0&0&0&0\\0&0&0&0&0&0\\1&1&0&0&0&0\\0&1&0&0&0&0\\1&0&1&0&0&0\\0&1&0&1&1&0\end{pmatrix}$.

Build the generalized-modeling community matrix at the unknown steady state from this topology. Row timescales are $\alpha_i=42^{-(t_i-1)/4}$ with Levine trophic levels $t_i$. Branching fractions come from $A$. The elasticities are $\phi=(0.22,0.38,0,0,0,0)$, $\gamma=(0.95,1.10,0.85,1.25,0.70,1.05)$, $\psi=(0.90,0.80,0.60,0.50,0.85,1.00)$, $\mu=(1.60,1.45,1.70,1.80,1.30,2.00)$, and $\lambda_{ij}=1$ except $\lambda_{65}=0.75$ and $\lambda_{51}=1.20$. Return the mass of the leading reactivity mode on the apparent-competition motif that has the largest reactivity.
In the reasoning, record the six trophic levels, the six timescales, the three-species sets that remain, the set that is discarded, the winning set, and the three squared mode weights that add to the tagged number. Also record the reactivity of the full web as a contrast. Do not paste $A$, the full Jacobian, or solver traces.

Output format requirements:

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:

The tags are required. Do not omit them or leave them empty.
The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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
import math
import numpy
import scipy.special
```

### Step 1

01_feeding_branching.py

Goal
----
From a binary feeding matrix, return consumer flags, prey flags, row-normalized diet shares, and column-normalized predator shares.

```python
def feeding_branching(A: "np.ndarray") -> tuple:
    '''Return rho, sigma, chi and beta from a binary feeding matrix.

    Parameters
    ----------
    A : np.ndarray
        Square array with A[i, j] = 1 if i consumes j, else 0.

    Returns
    -------
    rho : np.ndarray
        Shape (N,), 1.0 if row i has at least one prey.
    sigma : np.ndarray
        Shape (N,), 1.0 if column i has at least one predator.
    chi : np.ndarray
        Shape (N, N), row-normalized diet shares.
    beta : np.ndarray
        Shape (N, N), column-normalized predator shares.

    Raises
    ------
    ValueError
        If A is not a square 2-d array with N >= 1, or if any entry is not 0 or 1.
    '''
    return rho, sigma, chi, beta
```

### Step 2

02_row_timescales.py

Goal
----
From a feeding matrix, compute Levine trophic levels and return row timescales α_i=R^(-(t_i-1)/4).

```python
def row_timescales(A: "np.ndarray", R: float = 42.0) -> "np.ndarray":
    '''Return row timescales from Levine trophic levels.

    Parameters
    ----------
    A : np.ndarray
        Square feeding matrix, A[i, j] = 1 if i eats j.
    R : float
        Positive metabolic base. Default 42.

    Returns
    -------
    alpha : np.ndarray
        Shape (N,), alpha_i = R**(-(t_i-1)/4).

    Raises
    ------
    ValueError
        If A is not a square 2-d 0-1 array with N >= 1, if R is not finite
        and > 0, or if Levine iteration does not converge within 10000 steps.
    '''
    return alpha
```

### Step 3

03_community_matrix.py

Goal
----
Assemble the generalized-modeling community matrix from branching fractions, row timescales, and elasticities.

```python
def community_matrix(
    A: "np.ndarray",
    phi: "np.ndarray",
    gamma: "np.ndarray",
    psi: "np.ndarray",
    mu: "np.ndarray",
    lam: "np.ndarray",
    alpha: "np.ndarray",
    rho: "np.ndarray",
    sigma: "np.ndarray",
    chi: "np.ndarray",
    beta: "np.ndarray",
) -> "np.ndarray":
    '''Return the generalized-modeling community matrix J.

    Parameters
    ----------
    A : np.ndarray
        Square 0-1 feeding matrix.
    phi, gamma, psi, mu, alpha, rho, sigma : np.ndarray
        Length-N elasticity or scale vectors.
    lam, chi, beta : np.ndarray
        Shape (N, N) elasticity and branching matrices.

    Returns
    -------
    J : np.ndarray
        Shape (N, N) Jacobian at the unknown steady state.

    Raises
    ------
    ValueError
        If any array shape does not match N or (N, N), if A is not 0-1,
        or if any input is not finite.
    '''
    return J
```

### Step 4

04_reactivity.py

Goal
----
Return the reactivity of a square matrix. Refuse entries larger than 1e150 in absolute value.

```python
def reactivity(M: "np.ndarray") -> float:
    '''Return reactivity of a square matrix M.

    Parameters
    ----------
    M : np.ndarray
        Square real matrix, usually a Jacobian.

    Returns
    -------
    r : float
        Reactivity of M.

    Raises
    ------
    ValueError
        If M is not a square 2-d array with N >= 1, if any entry is not finite,
        if any absolute entry exceeds 1e150, or if the symmetric part is not finite.
    '''
    return r
```

### Step 5

05_apparent_competition_motifs.py

Goal
----
List every apparent-competition triad of a feeding matrix as a row [p, a, b] with a < b, rows sorted by (p, a, b).

```python
def apparent_competition_motifs(A: "np.ndarray") -> "np.ndarray":
    '''Return apparent-competition triads of a feeding matrix.

    Parameters
    ----------
    A : np.ndarray
        Square 0-1 feeding matrix, A[i, j] = 1 if i eats j.

    Returns
    -------
    motifs : np.ndarray
        Shape (n_motifs, 3), rows [p, a, b] with a < b, 0-based indices,
        rows sorted lexicographically by (p, a, b).
        Shape (0, 3) if none exist.

    Raises
    ------
    ValueError
        If A is not a square 2-d 0-1 array with N >= 1.
    '''
    return motifs
```

### Step 6

06_max_motif_reactivity.py

Goal
----
Given a square matrix and a nonempty motif index array, return the largest block reactivity among those rows. Indices in a row must be distinct.

```python
def max_motif_reactivity(S: "np.ndarray", motifs: "np.ndarray") -> float:
    '''Return the largest block reactivity among listed motifs.

    Parameters
    ----------
    S : np.ndarray
        Square matrix whose blocks are scored.
    motifs : np.ndarray
        Shape (n_motifs, k) integer node indices, 0-based.
        Each row must contain distinct indices.

    Returns
    -------
    r_motif : float
        Largest block reactivity over motif rows.

    Raises
    ------
    ValueError
        If S is not square and finite, if motifs is empty, if any index is
        out of range, or if a motif row contains duplicate indices.
    '''
    return r_motif
```

### Step 7

07_reactivity_mode_mass.py

Goal
----
Build the community matrix and return the mass of the leading reactivity mode on the apparent-competition motif with the largest reactivity.

```python
def reactivity_mode_mass(
    A: "np.ndarray",
    phi: "np.ndarray",
    gamma: "np.ndarray",
    psi: "np.ndarray",
    mu: "np.ndarray",
    lam: "np.ndarray",
    R: float = 42.0,
) -> float:
    '''Return the mass of the leading reactivity mode on the most reactive AC motif.

    Parameters
    ----------
    A : np.ndarray
        Square 0-1 feeding matrix, A[i, j] = 1 if i eats j.
    phi, gamma, psi, mu : np.ndarray
        Length-N elasticities.
    lam : np.ndarray
        Shape (N, N) diet elasticities.
    R : float
        Metabolic base for row timescales. Default 42.

    Returns
    -------
    mass : float
        Sum of squared entries of the unit leading reactivity mode
        on the winning apparent-competition motif.

    Raises
    ------
    ValueError
        If inputs are invalid, if Levine levels do not converge,
        if there is no apparent-competition motif, or if the symmetric part is not finite.
    '''
    return mass
```
