# Quantitative Analysis of Kinetic-Mechanism Complexity

## Background

Scientific Background

Enzyme function can depend nonlinearly on microscopic properties of a
multi-state catalytic process. Consequently, independent microscopic
perturbations can produce context-dependent effects in experimentally
observable kinetic quantities.

The supplied research source develops two representations of an enzyme
catalytic system and examines how changes in mechanistic complexity affect
the prevalence of apparent epistasis. This benchmark asks for a deterministic
numerical comparison of those representations using a fixed perturbation
ensemble.

## Problem

Using the supplied scientific source and the mutation perturbation matrix below, compute the ratio of the prevalence of significant catalytic-efficiency interactions in the complete representation to that in the reduced representation.

Sixteen independent mutation perturbations are supplied as rows of a matrix with six energetic components per row. The six columns correspond, in order, to the six microscopic states used by the complete representation. Energies are expressed in kcal mol^-1.

The mutation perturbation matrix is:

| Mutation | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | -1.142 | -0.587 | -0.808 | 0.183 | -1.197 | 0.377 |
| 2 | 1.721 | -0.897 | 0.146 | -0.720 | 0.754 | -1.816 |
| 3 | 0.193 | 0.857 | 0.419 | 0.660 | -0.231 | -0.347 |
| 4 | -0.432 | 0.164 | 1.439 | -0.331 | -0.945 | -1.063 |
| 5 | -0.302 | -0.704 | 1.026 | 1.832 | -1.826 | 0.627 |
| 6 | -1.272 | 0.357 | 1.791 | -0.544 | -1.307 | -0.301 |
| 7 | -1.608 | -1.751 | -0.370 | 0.899 | -1.649 | -0.648 |
| 8 | -1.608 | 1.134 | 1.278 | -1.465 | 0.324 | 0.171 |
| 9 | 0.378 | -0.635 | 1.920 | -1.307 | -0.705 | 1.545 |
| 10 | 1.555 | -0.150 | 0.982 | -0.368 | -1.020 | 0.925 |
| 11 | 1.965 | -1.692 | 1.529 | 0.764 | -1.538 | -1.641 |
| 12 | 1.084 | 1.988 | 1.739 | -1.844 | -0.143 | -1.299 |
| 13 | -1.649 | -1.613 | -0.919 | 0.515 | 0.051 | -0.791 |
| 14 | -1.127 | 0.295 | -0.330 | 0.863 | 1.577 | 0.269 |
| 15 | 0.767 | -1.968 | -0.942 | 1.776 | 1.157 | 1.189 |
| 16 | 0.666 | 0.974 | -1.131 | -1.801 | 1.619 | -1.009 |

Use the first four states for the reduced representation and all six states for the complete representation.

The wild-type free-energy vectors, in the corresponding state orderings, are:

- Reduced representation:
  `[0, 10, -5, 11]` kcal mol^-1
- Complete representation:
  `[0, 10, -5, 11, -9, 9]` kcal mol^-1

For both representations, construct the microscopic kinetic quantities from the state free energies using the transition-state-theory construction defined by the supplied scientific source. Use

- `R = 1.98720425864083e-3` kcal mol^-1 K^-1
- `T = 298.15` K
- `kB = 1.380649e-23` J K^-1
- `h = 6.62607015e-34` J s

Use the transition-state prefactor and energy convention specified by the source, with the energetic quantities expressed in kcal mol^-1 and the supplied value of `R`.

For each representation, obtain catalytic efficiency from the source-defined mapping between its microscopic kinetic rates and the macroscopic quantities `kcat` and `KM`. The reduced and complete representations have different kinetic-state structures, so use the appropriate source-defined mapping for each representation rather than applying one common kinetic expression to both.

For each of the sixteen mutations, obtain its perturbed state-energy vector by adding the corresponding row of the supplied perturbation matrix to the appropriate wild-type vector. For each unordered pair of distinct mutations, combine their perturbations additively in free-energy space and evaluate the corresponding double-mutant catalytic efficiency.

Construct the single-mutant and pairwise interaction quantities using the source-defined fold-change normalization and multiplicative null model. Apply the source's definition of a significant interaction, including its treatment of both directions of deviation from the null expectation. Do not replace the source convention with a one-sided or independently chosen threshold.

Evaluate every unordered pair of distinct mutations exactly once, giving

\[
\binom{16}{2}=120
\]

pairs for each representation. Count the source-defined significant interactions separately for the reduced and complete representations, divide each count by 120 to obtain its prevalence, and report the complete-representation prevalence divided by the reduced-representation prevalence.

Use binary64 arithmetic and retain full precision throughout the calculation. Do not round intermediate rate constants, kinetic quantities, interaction factors, or prevalence values.

The required result is the single prevalence ratio described above. The supplied perturbation matrix is the benchmark-specific numerical instance; determine all source-defined kinetic and significance conventions from the scientific source rather than assuming unstated alternative conventions.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.

You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
  Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
  lines.
- Keep <reasoning> short (a few hundred words). Show only the few scalars that
  determine the final number.

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

compute_tst_rates

Goal
----
Compute the microscopic transition-state-theory rate constants associated with a specified enzyme kinetic mechanism from its state free energies. The calculation must use the thermodynamic barrier between the initial state and transition state for each elementary transition and the common transition-state-theory prefactor. Support both the four-state mechanism containing k1, k−1, and k2 and the extended mechanism containing the additional k−2 and k3 transitions.

```python
def compute_tst_rates(
    energies: "np.ndarray",
    T: float,
    R: float,
    kB: float,
    h: float,
    complex_model: bool,
) -> "np.ndarray":
    """
    Convert microscopic state energies into the source-defined transition
    rates for the selected kinetic representation.

    Parameters
    ----------
    energies : np.ndarray
        Microscopic state-energy array.

    T : float
        Temperature used by the source-defined rate construction.

    R : float
        Gas constant in the units required by the supplied energies.

    kB : float
        Boltzmann constant in the units required by the rate construction.

    h : float
        Planck constant in the units required by the rate construction.

    complex_model : bool
        Selects the reduced or complete kinetic representation.

    Returns
    -------
    np.ndarray
        Microscopic rate constants in the representation selected by
        ``complex_model``.

    Raises
    ------
    ValueError
        If the energy array, thermodynamic constants, or representation
        selector is invalid.
    """
    return rates
```

### Step 2

compute_kinetic_observables

Goal
----
Map microscopic rate constants to the macroscopic kinetic observables used in the benchmark: kcat, KM, catalytic efficiency kcat/KM, and KD. The mapping must depend on whether the supplied rates describe the simple or extended catalytic cycle.

```python
def compute_kinetic_observables(
    rate_states: "np.ndarray",
    complex_model: bool,
) -> "np.ndarray":
    """
    Compute the source-defined kinetic observables from microscopic
    rate constants for the selected representation.

    Parameters
    ----------
    rate_states : np.ndarray
        Microscopic rate constants in the state ordering defined by the task.

    complex_model : bool
        Selects the reduced or complete kinetic representation.

    Returns
    -------
    np.ndarray
        A two-dimensional floating-point array with four observable
        columns in the ordering required by the downstream calculations.

    Raises
    ------
    ValueError
        If the rate array has an invalid shape, contains non-finite or
        non-positive values, or the representation selector is invalid.
    """
    return observables
```

### Step 3

build_additive_double_energies

Goal
----
Construct the microscopic free-energy configurations of all specified double mutants by combining the free-energy perturbations of two single mutants under the additive microscopic null model. Preserve the wild-type reference energies and generate one deterministic configuration for every requested mutant pair.

```python
def build_additive_double_energies(
    wild_type_energies: "np.ndarray",
    mutation_deltas: "np.ndarray",
    pair_indices: "np.ndarray",
) -> "np.ndarray":
    """
    Construct microscopic double-mutant state energies from the wild-type
    reference and the supplied mutation perturbations.

    Parameters
    ----------
    wild_type_energies : np.ndarray
        Wild-type microscopic state energies.

    mutation_deltas : np.ndarray
        Mutation perturbation vectors.

    pair_indices : np.ndarray
        Integer pairs identifying the two mutations to combine.

    Returns
    -------
    np.ndarray
        Microscopic energies for the requested double-mutant states.

    Raises
    ------
    ValueError
        If the wild-type energies, perturbation matrix, or pair indices
        have invalid shapes or contain invalid values.
    """
    return double_energies
```

### Step 4

compute_single_fold_changes

Goal
----
Normalize the kinetic observables of each single mutant by the corresponding wild-type kinetic observables to obtain dimensionless single-mutant fold changes. Return the fold changes for kcat, KM, catalytic efficiency, and KD using a fixed wild-type reference.

```python
def compute_single_fold_changes(
    observables: "np.ndarray",
    anchor_index: int,
) -> "np.ndarray":
    """
    Compute wild-type-normalized single-mutant fold changes.

    Parameters
    ----------
    observables : np.ndarray
        Observable matrix containing the reference state and mutant states.

    anchor_index : int
        Row index of the reference state.

    Returns
    -------
    np.ndarray
        Wild-type-normalized observable matrix.

    Raises
    ------
    ValueError
        If the observable matrix or reference index is invalid.
    """
    return fold_changes
```

### Step 5

compute_pair_interactions

Goal
----
Compute the macroscopic interaction factor for each double-mutant pair by comparing its kinetic observable obtained from the additive microscopic model with the conventional multiplicative null prediction constructed from the two single-mutant fold changes.

```python
def compute_pair_interactions(
    single_folds: "np.ndarray",
    anchor_observable: float,
    double_observables: "np.ndarray",
    pair_indices: "np.ndarray",
    anchor_index: int,
    observable_index: int,
) -> "np.ndarray":
    """
    Compute the source-defined interaction quantity for each supplied
    mutation pair.

    Parameters
    ----------
    single_folds : np.ndarray
        Wild-type-normalized single-mutant observables.

    anchor_observable : float
        Reference-state value of the observable being evaluated.

    double_observables : np.ndarray
        Observable values for the requested double mutants.

    pair_indices : np.ndarray
        Integer mutation-pair indices.

    anchor_index : int
        Row index of the reference mutation in ``single_folds``.

    observable_index : int
        Column identifying the kinetic observable to evaluate.

    Returns
    -------
    np.ndarray
        One positive interaction factor for every supplied mutation pair.

    Raises
    ------
    ValueError
        If the fold-change matrix, reference value, double-mutant
        observables, pair indices, reference index, or observable index
        is invalid.
    """
    return interactions
```

### Step 6

classify_epistasis

Goal
----
Determine the number and prevalence of significant double-mutant interactions from a vector of macroscopic interaction factors using the fixed 1.5-fold significance criterion. Separate significant interactions into positive and negative classes and report their aggregate counts and prevalence.

```python
def classify_epistasis(
    interactions: "np.ndarray",
    threshold: float,
) -> "np.ndarray":
    """
    Summarize significant pairwise interactions using the supplied
    significance threshold.

    Parameters
    ----------
    interactions : np.ndarray
        One-dimensional vector of interaction factors.

    threshold : float
        Multiplicative threshold used for significance classification.

    Returns
    -------
    np.ndarray
        A length-4 floating-point array containing, in order:
        [significant_count, positive_count, negative_count,
         significant_prevalence].

    Raises
    ------
    ValueError
        If the interaction vector is invalid or the threshold is invalid.
    """
    return summary
```

### Step 7

compute_complexity_ratio

Goal
----
Quantify the amplification of significant catalytic-efficiency epistasis produced by increasing the kinetic-cycle complexity. Given the significant-interaction prevalences obtained independently for the simple and extended mechanisms, return their dimensionless prevalence ratio.

```python
def compute_complexity_ratio(
    simple_stats: "np.ndarray",
    complete_stats: "np.ndarray",
) -> float:
    """
    Compute the prevalence ratio between the reduced and complete
    representations.

    Parameters
    ----------
    simple_stats : np.ndarray
        Length-4 summary array for the reduced representation.

    complete_stats : np.ndarray
        Length-4 summary array for the complete representation.

    Returns
    -------
    float
        Complete-representation significant-interaction prevalence divided
        by reduced-representation significant-interaction prevalence.

    Raises
    ------
    ValueError
        If either statistics array has an invalid shape or contains
        invalid values, or if the reduced prevalence is not positive.
    """
    return ratio
```

### Step 8

resolve_kinetic_complexity_ratio

Goal
----
Integrate the complete mechanistic workflow for both catalytic-cycle representations and return the deterministic amplification ratio of significant catalytic-efficiency epistasis. The integration must use the same wild-type energetic reference, mutation perturbations, pair construction, significance threshold, and pair ordering for both mechanisms so that their prevalence comparison isolates the effect of kinetic-cycle complexity.

```python
def resolve_kinetic_complexity_ratio(mutation_deltas: "np.ndarray") -> float:
    """Run both kinetic mechanisms over all 120 mutation pairs.

    Parameters
    ----------
    mutation_deltas : np.ndarray
        Six-component mutation free-energy perturbation matrix for the
        16 substitutions, with shape (16,6).

    Returns
    -------
    float
        Complete-mechanism significant-pair prevalence divided by the
        four-state significant-pair prevalence.

    Raises
    ------
    ValueError
        If mutation_deltas does not have shape (16,6) or contains
        non-finite values.
    """
    return ratio
```
