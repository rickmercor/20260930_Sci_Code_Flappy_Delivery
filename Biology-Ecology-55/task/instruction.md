# Biology-Ecology-55

## Background

Ecological interaction models can admit several alternative equilibrium communities under one fixed parameter set. Their number and species composition can both vary across parameter realizations, so biodiversity summaries must retain more structure than a single representative equilibrium.

Recent work represents each fixed realization by a finite distribution over its qualifying outcomes before lifting that object to a parameter ensemble. This retains system-level heterogeneity that ordinary one-state summaries discard.

For the paper's first dynamical case study, small systems can be evaluated deterministically from candidate community supports. The resulting ensemble richness statistic depends on the source's precise state and probability definitions.

## Problem

Multistable ecological systems can support different numbers and compositions of alternative communities, so an ensemble richness summary depends on the paper's choice of state object and sampling unit. For the finite ensemble below, evaluate the source-defined richness discrepancy.

Interpret each ordered pair consisting of `growth` and one interaction matrix as a parameter realization of the source's first dynamical case study. Define $E(p)$ exactly as the class in Definition 3.3, constructing it with Definitions 3.1 and 3.2 and the numerical hierarchy in Section 3.2; use `tol` as the numerical margin in every source-defined classification decision.

Use the following deterministic configuration:

- `interaction_matrices` =
  `[[[-1.000, -0.190, -1.970, -2.048], [-2.143, -1.000, -2.115, -0.598], [-2.132, -0.301, -1.000, -2.039], [-1.176, -0.312, -1.717, -1.000]],`
  ` [[-1.000,  0.000,  0.000,  0.000], [ 0.000, -1.000,  0.000,  0.000], [ 0.000,  0.000, -1.000,  0.000], [ 0.000,  0.000,  0.000, -1.000]],`
  ` [[-1.000, -1.400,  0.000,  0.000], [-1.300, -1.000,  0.000,  0.000], [ 0.000,  0.000, -1.000, -1.500], [ 0.000,  0.000, -1.200, -1.000]],`
  ` [[-1.000, -0.705, -1.304, -1.376], [-2.253, -1.000, -0.702, -0.570], [-1.216, -2.339, -1.000, -1.598], [-0.909, -1.757, -1.510, -1.000]],`
  ` [[-1.000, -0.914, -2.085, -0.513], [-0.613, -1.000, -1.965, -0.092], [-1.578, -1.251, -1.000, -1.194], [-0.374, -0.853, -1.392, -1.000]],`
  ` [[-1.000, -0.487, -0.474, -0.207], [-0.796, -1.000, -1.951, -0.719], [-1.105, -0.352, -1.000, -1.139], [-0.098, -1.496, -2.160, -1.000]]]`
- `growth` = `[1.0, 1.0, 1.0, 1.0]`, shared by all six systems
- `target_richness` = `2`
- `tol` = `1e-9` for all source-defined classification decisions

Treat the six realizations as the empirical parameter distribution. From the resulting $E(p)$ sets, apply Definitions 2.4 and 2.7 and evaluate the additive term on the right-hand side of Equation (13) at `target_richness`. Your final answer must be a single number: that source-defined richness-2 term. In <reasoning>, identify the source locations used and give only the minimal numerical checkpoints needed to audit the source-based computation.

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

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_01_enumerate_supports

Goal
----
A generalized Lotka-Volterra system with N species can have an equilibrium on any subset of the species pool. Exhaustive state characterization therefore begins with the 2^N binary support masks, including the all-zero extinction support. Ordering masks first by richness and then lexicographically makes the enumeration deterministic while preserving the one-to-one relation between a row and a candidate subcommunity.

```python
def enumerate_supports(n_species: int) -> np.ndarray:
    '''Enumerate every species support, including extinction.

    Parameters
    ----------
    n_species : int
        Positive number of species in the system.

    Raises
    ------
    ValueError
        If n_species is not a positive integer.

    Returns
    -------
    support_masks : np.ndarray
        Binary array of shape (2**n_species, n_species), ordered by richness
        and then lexicographically.
    '''
    return support_masks  # noqa: F821
```

### Step 2

02_02_solve_support_equilibria

Goal
----
For generalized Lotka-Volterra dynamics in vector form, dx/dt = diag(x)(r + A x), a candidate equilibrium on support I solves A_II x_I = -r_I and has zero abundance outside I. A support is admissible only when this restricted system is nonsingular and every resident abundance is strictly greater than the numerical tolerance. The empty support represents extinction and gives the zero vector; inadmissible supports are retained as all-NaN rows so later classifications stay aligned with the support enumeration.

```python
def solve_support_equilibria(
    interaction: np.ndarray,
    growth: np.ndarray,
    support_masks: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    '''Solve the GLV equilibrium equations on supplied supports.

    Parameters
    ----------
    interaction : np.ndarray
        Finite square interaction matrix A of shape (N, N).
    growth : np.ndarray
        Finite intrinsic growth vector r of shape (N,).
    support_masks : np.ndarray
        Binary array of candidate supports with shape (Q, N).
    tol : float
        Finite positive threshold below which a resident is not admissible.

    Raises
    ------
    ValueError
        If the matrix, vector, or support shapes are inconsistent, if an input
        contains non-finite values, if a support mask is not binary, or if tol
        is not finite and positive.

    Returns
    -------
    equilibria : np.ndarray
        Array of shape (Q, N); inadmissible supports are represented by all-NaN
        rows.
    '''
    return equilibria  # noqa: F821
```

### Step 3

03_03_compute_invasion_rates

Goal
----
At a generalized Lotka-Volterra state x, the per-capita growth vector is g(x) = r + A x. Entries on the support vanish at equilibrium, while entries outside the support are invasion growth rates. Non-positive invasion growth for every absent species is the saturation, or non-invasibility, condition. Rows that do not represent admissible candidate equilibria remain all-NaN so they cannot enter later classification.

```python
def compute_invasion_rates(
    interaction: np.ndarray,
    growth: np.ndarray,
    equilibria: np.ndarray,
) -> np.ndarray:
    '''Evaluate resident and absent-species per-capita growth rates.

    Parameters
    ----------
    interaction : np.ndarray
        Finite square GLV interaction matrix A.
    growth : np.ndarray
        Finite intrinsic growth vector r with one entry per species.
    equilibria : np.ndarray
        Candidate equilibria; each row must be entirely finite or entirely NaN.

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes, interaction or growth is
        non-finite, or an equilibrium row mixes finite and non-finite entries.

    Returns
    -------
    invasion_rates : np.ndarray
        Per-capita growth vectors, with all-NaN rows retained for rejected
        candidates.
    '''
    return invasion_rates  # noqa: F821
```

### Step 4

04_04_compute_spectral_abscissae

Goal
----
Local asymptotic stability of a generalized Lotka-Volterra equilibrium is determined by the Jacobian J(x) = diag(r + Ax) + diag(x)A. Its spectral abscissa is the largest real part among the eigenvalues. A candidate is asymptotically stable only when this scalar is strictly negative, while rejected supports retain NaN values for alignment with the exhaustive support list.

```python
def compute_spectral_abscissae(
    interaction: np.ndarray,
    growth: np.ndarray,
    equilibria: np.ndarray,
) -> np.ndarray:
    '''Compute the GLV Jacobian spectral abscissa at each candidate equilibrium.

    Parameters
    ----------
    interaction : np.ndarray
        Finite square interaction matrix A of shape (N, N).
    growth : np.ndarray
        Finite intrinsic growth vector r of shape (N,).
    equilibria : np.ndarray
        Array of shape (L, N), where each row is finite or entirely NaN.

    Raises
    ------
    ValueError
        If array shapes are incompatible, model parameters are non-finite, or
        an equilibrium row mixes finite and non-finite entries.

    Returns
    -------
    spectral_abscissae : np.ndarray
        Float array of length L. Rejected candidate rows produce NaN entries.
    '''
    return spectral_abscissae  # noqa: F821
```

### Step 5

05_05_classify_stable_states

Goal
----
A GLV candidate is retained as a stable state only when it is an admissible steady state, no absent species has a positive invasion rate, and the full ecological Jacobian has spectral abscissa below -tol. The strict spectral margin excludes neutral equilibria, while the non-invasibility test implements saturation before stability. Invalid support solutions remain ordinary zero classifications rather than disappearing from the support-indexed arrays.

```python
def classify_stable_states(
    support_masks: np.ndarray,
    equilibria: np.ndarray,
    invasion_rates: np.ndarray,
    spectral_abscissae: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    '''Classify support-restricted candidates as asymptotically stable states.

    Parameters
    ----------
    support_masks : np.ndarray
        Binary support array of shape (Q, N).
    equilibria : np.ndarray
        Candidate equilibria of shape (Q, N), with all-NaN invalid rows.
    invasion_rates : np.ndarray
        Per-capita growth vectors of shape (Q, N), aligned with equilibria.
    spectral_abscissae : np.ndarray
        Maximum real Jacobian eigenvalues of shape (Q,), with NaN for invalid
        candidates.
    tol : float
        Finite positive tolerance for abundance, residual, and stability tests.

    Raises
    ------
    ValueError
        If shapes are inconsistent, masks are non-binary, finite and NaN rows
        are misaligned, or tol is not finite and positive.

    Returns
    -------
    stable_flags : np.ndarray
        Integer array of shape (Q,), equal to one exactly for stable states.
    '''
    return stable_flags  # noqa: F821
```

### Step 6

06_06_summarize_system_richness

Goal
----
For one parameterized ecological system p, the stable-state multiplicity S_p is the total number of stable equilibria, and S_p^k counts those equilibria whose support contains exactly k species. These counts retain the identity of the system that generated the states. The compact summary [S_p, S_p^0, ..., S_p^N] is the deterministic input to the system-centered probability construction.

```python
def summarize_system_richness(
    support_masks: np.ndarray,
    stable_flags: np.ndarray,
) -> np.ndarray:
    '''Count stable equilibria by support richness for one GLV system.

    Parameters
    ----------
    support_masks : np.ndarray
        Binary matrix with one species support per row.
    stable_flags : np.ndarray
        Binary vector marking the stable states.

    Raises
    ------
    ValueError
        If support_masks is not a nonempty binary matrix, or stable_flags does
        not contain one binary value per support.

    Returns
    -------
    system_summary : np.ndarray
        Integer vector [S_p, S_p^0, ..., S_p^N].
    '''
    return system_summary  # noqa: F821
```

### Step 7

07_07_build_system_probability_vectors

Goal
----
The system-centered probability vector F_p has entries indexed by the no-state outcome and richness values 0 through N. If a system has no retained state, all probability mass is assigned to the no-state entry. Otherwise, the no-state entry is zero and each richness entry is S_p^k/S_p, so every system with states contributes one normalized distribution regardless of its multiplicity.

```python
def build_system_probability_vectors(system_summaries: np.ndarray) -> np.ndarray:
    '''Normalize richness counts within each ecological system.

    Parameters
    ----------
    system_summaries : np.ndarray
        Non-negative integer array of shape (M, N + 2), with each row equal
        to (S_p, S_p^0, ..., S_p^N).

    Raises
    ------
    ValueError
        If the input is not a two-dimensional non-negative integer array with
        at least one richness column, or S_p differs from the richness-count sum.

    Returns
    -------
    probability_vectors : np.ndarray
        Float array of shape (M, N + 2), with columns for no state followed by
        richness values 0 through N.
    '''
    return probability_vectors  # noqa: F821
```

### Step 8

08_08_compare_richness_weightings

Goal
----
Conditioned on systems with S_p > 0, the system-centered richness probability is the mean of F_p^k across systems, while the state-centered probability pools all states before normalizing. These weight systems differently whenever multiplicity covaries with within-system richness composition. The signed pooling bias at richness k is p_state^k - p_system^k, positive when high-multiplicity systems disproportionately contribute states of that richness.

```python
def compare_richness_weightings(
    system_summaries: np.ndarray,
    probability_vectors: np.ndarray,
    target_richness: int,
) -> np.ndarray:
    '''Compare system-centered and pooled-state richness probabilities.

    Parameters
    ----------
    system_summaries : np.ndarray
        Nonnegative integer rows [S_p, S_p^0, ..., S_p^N].
    probability_vectors : np.ndarray
        Probability rows [F_p^empty, F_p^0, ..., F_p^N] derived from the
        supplied summaries.
    target_richness : int
        Richness k between zero and N, inclusive.

    Raises
    ------
    ValueError
        If inputs are inconsistent, target_richness is outside 0 through N,
        probability rows are invalid, or no system has a stable state.

    Returns
    -------
    comparison : np.ndarray
        Float vector [p_system^k, p_state^k, pooling_bias].
    '''
    return comparison  # noqa: F821
```

### Step 9

09_09_run_full_pipeline

Goal
----
The full pipeline enumerates every support once, solves and classifies all equilibrium candidates for each interaction matrix, summarizes stable-state multiplicity by richness, constructs one normalized probability vector per system, and compares system-centered with pooled-state weighting. The returned scalar is the signed excess p_state^k - p_system^k at target_richness k. This file is the orchestrator and is the step marked final in Studio.

```python
def run_full_pipeline(
    interaction_matrices: np.ndarray,
    growth: np.ndarray,
    target_richness: int = 2,
    tol: float = 1e-9,
) -> float:
    '''Run the stable-state ensemble analysis and return the pooling bias.

    Parameters
    ----------
    interaction_matrices : np.ndarray
        Finite array of shape (n_systems, n_species, n_species).
    growth : np.ndarray
        Shared finite intrinsic growth vector with one entry per species.
    target_richness : int
        Richness k between zero and n_species, inclusive.
    tol : float
        Finite positive tolerance for feasibility and stability decisions.

    Raises
    ------
    ValueError
        If the ensemble is empty or has an invalid shape, growth is
        incompatible, target_richness is invalid, or tol is not positive.

    Returns
    -------
    pooling_bias : float
        Signed excess of pooled-state over system-centered probability at k.
    '''
    return pooling_bias  # noqa: F821
```
