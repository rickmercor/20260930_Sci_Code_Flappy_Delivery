# debye_huckel_mean_activity_coefficient

## Background

SISR represents each elementary reaction using paired reactant and product stoichiometric vectors. Their difference supplies the signed contribution to every species balance, while the reactant vector determines the corresponding mass-action concentration monomial. This constrains the search to chemically interpretable polynomial kinetics.

For a fixed mechanism, SISR differentiates the measured concentration series and fits all rate constants together in derivative space. Each species derivative is scaled before forming the residual so that a rapidly changing species cannot dominate the fit. Candidate mechanisms are ranked by derivative error during the search, but the final scientific choice is not simply the mechanism with the smallest derivative residual.

The final comparison integrates each fitted mechanism and measures its error against separately scaled concentration trajectories. It then balances that error against an expression-tree complexity that penalizes additional kinetic terms, nonlinear reactant products, and larger stoichiometric coefficients without penalizing the numerical magnitude of a fitted rate constant. This distinction matters here because an additional reaction can absorb small perturbations and improve both numerical losses while still providing a less defensible mechanistic explanation.

## Problem

A reaction system contains four measured species in the order A, B, C, D. The table below gives a lightly perturbed concentration time series at nonuniform sampling times.

| time | A | B | C | D |
|---:|---:|---:|---:|---:|
| 0.00 | 1.20000000 | 0.08000000 | 0.03000000 | 0.00000000 |
| 0.07 | 1.19680873 | 0.08504694 | 0.03235152 | 0.00034691 |
| 0.16 | 1.17638285 | 0.09311654 | 0.03517167 | 0.00083757 |
| 0.31 | 1.16569019 | 0.10653083 | 0.04093369 | 0.00174251 |
| 0.52 | 1.12434706 | 0.12663981 | 0.05090539 | 0.00329966 |
| 0.80 | 1.08242928 | 0.15862843 | 0.06644897 | 0.00592546 |
| 1.15 | 0.99813266 | 0.20541817 | 0.09329393 | 0.01029542 |
| 1.58 | 0.89449655 | 0.26208953 | 0.13680727 | 0.01812513 |
| 2.10 | 0.74625165 | 0.32244038 | 0.20655167 | 0.03241652 |
| 2.75 | 0.58036819 | 0.35342079 | 0.31732050 | 0.05953419 |
| 3.55 | 0.42378397 | 0.31233118 | 0.46467954 | 0.10945823 |
| 4.55 | 0.31666019 | 0.20448149 | 0.59276695 | 0.19600106 |
| 5.80 | 0.25529348 | 0.09801058 | 0.63728962 | 0.32122890 |
| 7.40 | 0.22857335 | 0.03643595 | 0.56993106 | 0.47468379 |
| 9.40 | 0.21591196 | 0.01192926 | 0.44379018 | 0.63892659 |
| 12.00 | 0.21277023 | 0.00358169 | 0.30327429 | 0.78970170 |

The reaction library is:

R1: A + B -> 2 B
R2: B -> C
R3: B + C -> 2 C
R4: C -> D
R5: A -> B
R6: A + B -> B + C
R7: B -> D
R8: A -> D
R9: 2 C -> C + D
R10: B + D -> C

Evaluate these candidate mechanisms:

Candidate 1: R1, R2, R4
Candidate 2: R1, R2, R3, R4
Candidate 3: R1, R2, R3, R4, R10
Candidate 4: R5, R2, R3, R4
Candidate 5: R6, R2, R4
Candidate 6: R1, R3, R4
Candidate 7: R1, R2, R3, R9
Candidate 8: R1, R2, R3, R4, R7

Apply the stoichiometrically-informed symbolic regression (SISR) construction from the source paper. Fit all nonnegative rate constants for each candidate simultaneously in normalized derivative space. Evaluate the fitted mechanisms in concentration space from the first measured row and calculate the paper's expression-tree complexity.

For this deterministic benchmark, form the nondominated front in (complexity, log10 concentration loss). Sort it by increasing complexity, scale both coordinates to [0, 1], and select the interior point with the greatest perpendicular distance from the chord joining the two endpoints. If the front has fewer than three points, select its lowest-loss member. This chord rule is a local benchmark convention that makes the paper's qualitative elbow choice reproducible.

Return the 1-based candidate index selected by this procedure as a single numeric value.

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
Return a single Python float: the mean ionic activity coefficient gamma_pm (dimensionless, in the interval (0, 1] for these inputs). At infinite dilution (I = 0) the function must return exactly 1.0.
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

estimate_derivatives

Goal
----
Estimate concentration derivatives on a nonuniform time grid.

```python
def estimate_derivatives(times: np.ndarray, concentrations: np.ndarray) -> np.ndarray:
    """Estimate concentration derivatives on a nonuniform time grid.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing one-dimensional sampling times.
    concentrations : np.ndarray
        Concentrations with shape (n_times, n_species).

    Returns
    -------
    np.ndarray
        Derivative estimates with the same shape as concentrations.
    """
    return result
```

### Step 2

mass_action_features

Goal
----
Build mass-action monomial features from reactant stoichiometry.

```python
def mass_action_features(concentrations: np.ndarray, reactions: np.ndarray) -> np.ndarray:
    """Build mass-action monomial features from reactant stoichiometry.

    Parameters
    ----------
    concentrations : np.ndarray
        Nonnegative array with shape (n_times, n_species).
    reactions : np.ndarray
        Integer rows in [reactants | products] form.

    Returns
    -------
    np.ndarray
        Feature matrix with shape (n_times, n_reactions).
    """
    return result
```

### Step 3

stoichiometric_design

Goal
----
Convert reaction features into species-balance design columns.

```python
def stoichiometric_design(
    features: np.ndarray,
    reactions: np.ndarray,
    derivative_scales: np.ndarray,
) -> np.ndarray:
    """Convert reaction features into species-balance design columns.

    Parameters
    ----------
    features : np.ndarray
        Mass-action features with shape (n_times, n_reactions).
    reactions : np.ndarray
        Integer rows in [reactants | products] form.
    derivative_scales : np.ndarray
        Positive maximum derivative magnitude for each species.

    Returns
    -------
    np.ndarray
        Time-major design matrix with shape (n_times*n_species, n_reactions).
    """
    return result
```

### Step 4

fit_rate_constants

Goal
----
Fit nonnegative rate constants in normalized derivative space.

```python
def fit_rate_constants(
    design: np.ndarray,
    derivatives: np.ndarray,
    derivative_scales: np.ndarray,
) -> np.ndarray:
    """Fit nonnegative rate constants in normalized derivative space.

    Parameters
    ----------
    design : np.ndarray
        Species-scaled design matrix in time-major row order.
    derivatives : np.ndarray
        Measured derivatives with shape (n_times, n_species).
    derivative_scales : np.ndarray
        Positive scale for each species.

    Returns
    -------
    np.ndarray
        Fitted rates followed by the normalized derivative loss.
    """
    return result
```

### Step 5

concentration_loss

Goal
----
Score a fitted mechanism by integrating its ODE against concentration data.

```python
def concentration_loss(
    times: np.ndarray,
    initial_concentrations: np.ndarray,
    observed_concentrations: np.ndarray,
    reactions: np.ndarray,
    rates: np.ndarray,
    concentration_scales: np.ndarray,
) -> float:
    """Score a fitted mechanism by integrating its ODE against concentration data.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing sampling times.
    initial_concentrations : np.ndarray
        Initial species concentrations.
    observed_concentrations : np.ndarray
        Observed concentrations with shape (n_times, n_species).
    reactions : np.ndarray
        Integer rows in [reactants | products] form.
    rates : np.ndarray
        Nonnegative rate constants, one per reaction.
    concentration_scales : np.ndarray
        Positive scale for each species.

    Returns
    -------
    float
        Mean scaled concentration-trajectory loss.
    """
    return result
```

### Step 6

expression_complexity

Goal
----
Measure the symbolic complexity of a candidate reaction mechanism.

```python
def expression_complexity(reactions: np.ndarray) -> float:
    """Measure the symbolic complexity of a candidate reaction mechanism.

    Parameters
    ----------
    reactions : np.ndarray
        Integer rows in [reactants | products] form.

    Returns
    -------
    float
        Complexity score for the mechanism.
    """
    return result
```

### Step 7

pareto_front

Goal
----
Identify candidates that are not dominated in loss-complexity space.

```python
def pareto_front(losses: np.ndarray, complexities: np.ndarray) -> np.ndarray:
    """Identify candidates that are not dominated in loss-complexity space.

    Parameters
    ----------
    losses : np.ndarray
        Concentration-trajectory losses for candidate mechanisms.
    complexities : np.ndarray
        Complexity scores for the same candidates.

    Returns
    -------
    np.ndarray
        Boolean mask marking Pareto candidates.
    """
    return result
```

### Step 8

discover_sisr_mechanism

Goal
----
Run the SISR-style mechanism discovery pipeline and return the selected candidate.

```python
def discover_sisr_mechanism(
    times: np.ndarray,
    concentrations: np.ndarray,
    candidate_reactions: list,
) -> int:
    """Run the SISR-style pipeline and select the final mechanism.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing sampling times.
    concentrations : np.ndarray
        Observed concentrations with shape (n_times, n_species).
    candidate_reactions : list
        Candidate mechanisms in [reactants | products] form.

    Returns
    -------
    int
        One-based index of the selected candidate.
    """
    return result
```
